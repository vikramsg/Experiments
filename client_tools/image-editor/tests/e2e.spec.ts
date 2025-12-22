import { expect, test, type Page } from '@playwright/test';
import { PDFDocument, StandardFonts, rgb } from 'pdf-lib';

const redSvg = `<?xml version="1.0"?>
<svg xmlns="http://www.w3.org/2000/svg" width="300" height="200">
  <rect width="100%" height="100%" fill="#dc2626" />
</svg>`;

const blueSvg = `<?xml version="1.0"?>
<svg xmlns="http://www.w3.org/2000/svg" width="260" height="160">
  <rect width="100%" height="100%" fill="#2563eb" />
</svg>`;

const disableAnimations = async (page: Page) => {
  await page.addStyleTag({
    content: `* { transition: none !important; animation: none !important; }`
  });
};

const buildSamplePdf = async () => {
  const pdfDoc = await PDFDocument.create();
  const page = pdfDoc.addPage([420, 280]);
  const font = await pdfDoc.embedFont(StandardFonts.HelveticaBold);

  page.drawText('PDF Test File', {
    x: 40,
    y: 180,
    size: 28,
    font,
    color: rgb(0.2, 0.2, 0.2)
  });

  page.drawRectangle({
    x: 40,
    y: 60,
    width: 340,
    height: 90,
    borderColor: rgb(0.37, 0.52, 0.86),
    borderWidth: 3
  });

  const bytes = await pdfDoc.save();
  return Buffer.from(bytes);
};

test('compositor: upload, crop mode, screenshot', async ({ page }) => {
  await page.goto('/');
  await disableAnimations(page);

  await page.setInputFiles('input[type="file"]', [
    {
      name: 'base.svg',
      mimeType: 'image/svg+xml',
      buffer: Buffer.from(redSvg)
    },
    {
      name: 'overlay.svg',
      mimeType: 'image/svg+xml',
      buffer: Buffer.from(blueSvg)
    }
  ]);

  await expect(page.getByText('BASE LAYER')).toBeVisible();
  await expect(page.getByText('overlay')).toBeVisible();

  await page.getByTitle('Crop Tool').click();
  await expect(page.getByText(/CROP MODE ACTIVE/i)).toBeVisible();

  const eastHandle = page.locator('.cursor-e-resize').first();
  const handleBox = await eastHandle.boundingBox();
  if (!handleBox) throw new Error('Crop handle not found');

  await page.mouse.move(handleBox.x + handleBox.width / 2, handleBox.y + handleBox.height / 2);
  await page.mouse.down();
  await page.mouse.move(handleBox.x - 40, handleBox.y + handleBox.height / 2);
  await page.mouse.up();

  await expect(page.getByTestId('compositor-workspace')).toHaveScreenshot('compositor-crop.png');
});

test('single: upload, crop, export, screenshot', async ({ page }) => {
  await page.goto('/');
  await disableAnimations(page);

  await page.getByRole('button', { name: 'Single Image' }).click();

  await page.setInputFiles('input[type="file"]', {
    name: 'single.svg',
    mimeType: 'image/svg+xml',
    buffer: Buffer.from(redSvg)
  });

  await expect(page.getByText(/SINGLE CROP MODE/i)).toBeVisible();

  const westHandle = page.locator('.cursor-w-resize').first();
  const handleBox = await westHandle.boundingBox();
  if (!handleBox) throw new Error('Crop handle not found');

  await page.mouse.move(handleBox.x + handleBox.width / 2, handleBox.y + handleBox.height / 2);
  await page.mouse.down();
  await page.mouse.move(handleBox.x + 40, handleBox.y + handleBox.height / 2);
  await page.mouse.up();

  const downloadPromise = page.waitForEvent('download');
  await page.getByRole('button', { name: /export/i }).click();
  await downloadPromise;

  await expect(page.getByTestId('single-workspace')).toHaveScreenshot('single-crop.png');
});

test('pdf: upload, describe, compress, download', async ({ page }) => {
  test.setTimeout(60_000);
  await page.goto('/');
  await disableAnimations(page);

  await page.getByRole('button', { name: 'PDF Tools' }).click();
  await page.setInputFiles('input[type="file"]', {
    name: 'sample.pdf',
    mimeType: 'application/pdf',
    buffer: await buildSamplePdf()
  });

  await expect(page.getByTestId('pdf-preview')).toBeVisible();
  await page.getByRole('button', { name: /describe pdf/i }).click();
  await expect(page.getByText('Pages')).toBeVisible();

  await page.getByRole('button', { name: /compress pdf/i }).click();
  await expect(page.getByRole('button', { name: /download compressed/i })).toBeVisible({ timeout: 30000 });

  const downloadPromise = page.waitForEvent('download');
  await page.getByRole('button', { name: /download compressed/i }).click();
  await downloadPromise;
});
