
import pyreadstat

# this df will have the original values
df, meta = pyreadstat.read_sas7bdat('/Users/ruoxiliu/Desktop/GSS_sas/gss7222_r4.sas7bdat')
# read_sas7bdat returns an emtpy data frame and the catalog
df_empty, catalog = pyreadstat.read_sas7bdat('/Users/ruoxiliu/Desktop/GSS_sas/formats.sas7bcat')
# enrich the dataframe with the catalog
# formats_as_category is by default True, and it means the replaced values will be transformed to a pandas category column. formats_as_ordered_category is by default False meaning by default categories are not ordered.
df_enriched, meta_enriched = pyreadstat.set_catalog_to_sas(df, meta, catalog, 
                             formats_as_category=True, formats_as_ordered_category=False)