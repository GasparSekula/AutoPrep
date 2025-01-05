from typing import List

from pylatex import (
    Center,
    Command,
    Document,
    Figure,
    NoEscape,
    Section,
    Subsection,
    Table,
    Tabular,
)
from pylatex.package import Package

from ..utils.config import config
from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)


class Report:
    """
    Generates LaTeX reports with analysis results and visualizations.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Report, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        logger.info("Initializing ReportGenerator with title: %s", config.raport_title)

        try:
            self.doc = Document(
                default_filepath=config.root_dir, geometry_options=config.tex_geomatry
            )
            self.title = config.raport_title

            # Add necessary packages
            self.doc.packages.append(Package("graphicx"))
            self.doc.packages.append(Package("float"))
            self.doc.packages.append(Package("booktabs"))
            self.doc.packages.append(Package("hyperref"))
            self.doc.packages.append(Package("caption"))
            self.doc.packages.append(Package("subcaption"))
            logger.debug("Initialized LaTeX document with packages")
        except Exception as e:
            logger.error("Failed to initialize ReportGenerator: %s", str(e))
            raise

    def add_header(self) -> None:
        """
        Adds the title page and table of contents to the LaTeX document.
        """
        try:
            logger.debug("Initializing header.")
            self.doc.preamble.append(NoEscape(r"\title{" + config.raport_title + "}"))
            self.doc.preamble.append(NoEscape(r"\author{" + config.raport_author + "}"))
            self.doc.preamble.append(NoEscape(r"\date{\today}"))
            self.doc.append(NoEscape(r"\maketitle"))
            self.doc.append(config.raport_abstract)
            self.doc.append(Command("tableofcontents"))
            self.doc.append(NoEscape(r"\newpage"))
        except Exception as e:
            logger.error("Failed to add header: %s", str(e))
            raise

    def add_section(self, title: str, description: str = "") -> Section:
        """
        Adds a new section to the LaTeX document.

        Args:
            title (str): The title of the section.
            description (str, optional): The description of the section. Defaults to "".

        Returns:
            Section: The newly created section.
        """
        section = Section(title)
        if description:
            section.append(description)
        self.doc.append(section)
        return section

    def add_subsection(self, title: str) -> Subsection:
        """
        Adds a new subsection to the LaTeX document.

        Args:
            title (str): The title of the subsection.

        Returns:
            Subsection: The newly created subsection.
        """
        subsection = Subsection(title)
        self.doc.append(subsection)
        return subsection

    def add_table(
        self,
        data: dict | list,
        caption: str = None,
        header: List[dict] = ["Category", "Value"],
    ) -> None:
        """Add a table to the document.

        Args:
            data (dict | list): Data to convert to table.
            caption (str): Table caption. If None, no caption will be set.
            header (List[dict]): Table header. Defaults to ["Category", "Value"].
                If None, no header will be set.
        """
        decimal_precision = config.raport_decimal_precision

        try:
            if isinstance(data, dict):
                data = list(data.items())

            with self.doc.create(Table(position="H")) as table:
                with table.create(Center()) as centered:
                    # Determine the number of columns
                    num_columns = len(header) if header is not None else len(data[0])
                    # Generate column alignment string (e.g., "l r c" for 3 columns)
                    alignment = " ".join(["l" for _ in range(num_columns)])

                    with centered.create(Tabular(alignment)) as tabular:
                        tabular.add_hline()
                        if header is not None:
                            if len(header) != num_columns:
                                raise ValueError(
                                    f"Header length ({len(header)}) does not match "
                                    f"number of columns ({num_columns})."
                                )
                            tabular.add_row(
                                [NoEscape(f"\\textbf{{{c}}}") for c in header]
                            )
                            tabular.add_hline()

                        for row in data:
                            # Check row length matches the number of columns
                            if len(row) != num_columns:
                                raise ValueError(
                                    f"Row length ({len(row)}) does not match number of "
                                    f"columns ({num_columns}): {row}"
                                )

                            # Format row elements
                            formatted_row = [
                                (
                                    f"{value:.{decimal_precision}f}"
                                    if isinstance(value, float)
                                    else str(value)
                                )
                                for value in row
                            ]
                            tabular.add_row(formatted_row)

                        tabular.add_hline()

                if caption is not None:
                    table.add_caption(caption)
        except Exception as e:
            logger.error(f"Failed to add table {caption}: {str(e)}")
            raise

    def add_figure(self, path: str, caption: str = None, size: int = 0.9) -> None:
        """Add a figure to the document.

        Args:
            path (str): Path to the figure file.
            caption (str): Figure caption. If None, no caption will be set.
            size (int): % of width image is to take. Defaults to 0.9.
        """
        try:
            with self.doc.create(Figure(position="H")) as fig:
                fig.add_image(path, width=NoEscape(rf"{size}\textwidth"))
                if caption is not None:
                    fig.add_caption(caption)
        except Exception as e:
            logger.error(f"Failed to add figure {path}: {str(e)}")
            raise

    def add_verbatim(self, content: str) -> str:
        """Add verbatim text to the document.

        Args:
            content (str): Text to add in verbatim environment.

        Returns:
            str: Formatted text.
        """
        try:
            self.doc.append(
                NoEscape(
                    r"\begin{verbatim}" + "\n" + content + "\n" + r"\end{verbatim}"
                )
            )
            return content
        except Exception as e:
            logger.error("Failed to add verbatim content: %s", str(e))
            raise

    def generate(self) -> None:
        """
        Generate the final PDF report.

        Args:
            output_path (str): Path where to save the PDF.
        """
        try:
            logger.debug(f"Generating PDF at {config.raport_path}")
            self.doc.generate_pdf(
                config.raport_path, clean=False, clean_tex=False, compiler="pdflatex"
            )
            self.doc.generate_pdf(
                config.raport_path, clean_tex=config.return_tex_, compiler="pdflatex"
            )
            logger.info("PDF generation complete")
        except Exception as e:
            logger.error(f"Failed to generate PDF: {str(e)}")
            raise


raport = Report()
