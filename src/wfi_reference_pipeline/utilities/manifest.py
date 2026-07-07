import pandas as pd
from roman_datamodels import datamodels as rdd

# ----------------------------------------------------------------------
# Define all metadata fields you want to collect here and check for
# when creating the manifest. Each line can be a single instance in a
# ref type that has a unique meta field name.
#
# The dictionary key becomes the DataFrame column name.
# The tuple is the path through the metadata tree.
# ----------------------------------------------------------------------

META_FIELDS = {
    "reftype": ("reftype",),
    "author": ("author",),
    "description": ("description",),
    "pedigree": ("pedigree",),
    "origin": ("origin",),
    "telescope": ("telescope",),
    "useafter": ("useafter",),
    "detector": ("instrument", "detector"),
    "instrument_name": ("instrument", "name"),
    "optical_element": ("instrument", "optical_element"),
    "exptype": ("exposure", "type"),
    "p_exptype": ("exposure", "p_exptype"),
    "input_units": ("input_units",),
    "output_units": ("output_units",),
}


def get_nested_value(meta, path, default="N/A"):
    """
    Safely retrieve a nested metadata value.

    Parameters
    ----------
    meta : dict-like
        Roman metadata tree.
    path : tuple
        Tuple describing the path to the desired value.
    default : object
        Value returned if the key doesn't exist.

    Returns
    -------
    object
        Metadata value or default.
    """
    value = meta

    for key in path:
        try:
            value = value[key]
        except (KeyError, TypeError):
            return default

    return value


def make_manifest(files):
    """
    Create a pandas DataFrame containing selected metadata from a list of
    Roman reference files.

    Parameters
    ----------
    files : list[str]
        List of ASDF reference files.

    Returns
    -------
    pandas.DataFrame
        One row per reference file and one column for each metadata field
        defined in ``META_FIELDS``.

    Examples
    --------
    Create a manifest from all ASDF files in the current directory::

    from wfi_reference_pipeline.utilities.manifest import make_manifest, print_manifest, print_meta_fields_together
    import glob, asdf
    files = glob.glob("*.asdf")
    manifest = make_manifest(files)

    View the DataFrame::
    print(manifest)

    Print one file's metadata at a time::
    print_manifest(manifest)

    Print values grouped by metadata field::
    print_meta_fields_together(manifest)
    """

    files.sort()
    rows = []
    for filename in files:
        with rdd.open(filename) as rf:
            meta = rf.meta

        row = {"file": filename}
        for column, path in META_FIELDS.items():
            value = get_nested_value(meta, path)
            # Convert Time objects into readable strings
            if column == "useafter" and value != "N/A":
                value = value.datetime.strftime("%Y-%m-%d %H:%M:%S")
            # Convert long lists into something printable
            elif isinstance(value, list):
                value = ", ".join(str(v) for v in value)

            row[column] = value

        rows.append(row)

    df = pd.DataFrame(rows)

    return df

def print_manifest(df):
    """
    Print metadata for each file.



    """
    for _, row in df.iterrows():
        print(f"File: {row['file']}")
        for column in df.columns:
            if column == "file":
                continue
            print(f"{column}: {row[column]}")
        print("-" * 60)

def print_meta_fields_together(df):
    """
    Print each metadata field grouped together.
    """
    for column in df.columns:
        print(f"{column}")
        for value in df[column]:
            print(f"  {value}")
        print("-" * 60)