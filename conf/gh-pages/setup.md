# MkDocs GitHub Pages Deployment

## Prerequisites

- A GitHub repository (must be **public** for free GitHub Pages)
- MkDocs project with documentation in the `docs/` folder

## Project Structure

```
.
├── .github/workflows/mkdocs.yml    # GitHub Action workflow
├── docs/
│   └── index.md                    # Your documentation
├── mkdocs.yml                      # MkDocs configuration
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Step-by-Step Setup

### 1. Setup docs and mkdocs

Make sure anything required for `mkdocs` is locally setup. This includes

1. mkdocs.yml
2. requirements.txt - including dependencies for mkdocs plugins.

### 2. Configure Repository Settings

**Critical Step**: The GitHub Action builds and deploys your site, but you must manually configure GitHub Pages settings:

1. Go to your repository on GitHub
2. Navigate to **Settings** → **Pages** (in the left sidebar)
3. Under **"Build and deployment"**:
   - **Source**: Select "Deploy from a branch"
   - **Branch**: Select `gh-pages` 
   - **Folder**: Select `/ (root)`
4. Click **"Save"**

### 3. Make Repository Public

GitHub Pages requires a public repository (unless you have GitHub Pro/Enterprise):

1. Go to **Settings** → **General**
2. Scroll to **"Danger Zone"**
3. Click **"Change repository visibility"**
4. Select **"Make public"**

### 4. Add Website to Repository About Section

Make your site easily discoverable:

1. Go to your repository's main page on GitHub
2. Click the **gear icon (⚙️)** next to "About" on the right sidebar
3. Check **"Use your GitHub Pages website"** 
4. Click **"Save changes"**

This adds a clickable link to your documentation site directly on your repository page.

## Deployment Process

### How It Works

1. **Trigger**: Push to `main` branch triggers the GitHub Action
2. **Build**: Action installs dependencies and runs `mkdocs build`
3. **Deploy**: Built site is pushed to `gh-pages` branch
4. **Serve**: GitHub Pages serves the site from `gh-pages` branch

### Checking Deployment Status

After pushing changes, monitor deployment:

#### Actions Tab
1. Go to your repository's **"Actions"** tab
2. Look for the latest "Publish docs via GitHub Pages" workflow run
3. Check if it completed successfully (green checkmark)


## Accessing Your Site

Once deployed, your documentation will be available at:
```
https://[username].github.io/[repository-name]/
```

