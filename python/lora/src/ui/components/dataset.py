"""Dataset and record components."""

from fasthtml.common import H2, H3, Audio, Div, Table, Tbody, Td, Th, Thead, Tr


def DatasetTable(datasets):
    """Render a list of datasets as a table."""
    rows = []
    for ds in datasets:
        rows.append(
            Tr(
                Td(ds.id),
                Td(ds.name),
                Td(ds.description or "No description"),
                Td(str(ds.created_at.date())),
                Td(str(len(ds.records))),
            )
        )

    return Div(
        H2("Available Datasets"),
        Table(
            Thead(Tr(Th("ID"), Th("Name"), Th("Description"), Th("Created"), Th("Records"))),
            Tbody(*rows),
            cls="striped",
        ),
    )


def RecordTable(records):
    """Render a list of records as a table."""
    rows = []
    for rec in records:
        content_cell = Td(rec.content)
        audio_cell = (
            Td(Audio(src=f"/audio/{rec.id}", controls=True))
            if rec.data_type == "AUDIO"
            else Td("-")
        )
        rows.append(
            Tr(
                Td(rec.id),
                Td(rec.data_type),
                content_cell,
                audio_cell,
            )
        )

    return Div(
        H3("Records Preview"),
        Table(
            Thead(Tr(Th("ID"), Th("Type"), Th("Content"), Th("Preview"))),
            Tbody(*rows),
            cls="striped",
        ),
    )
