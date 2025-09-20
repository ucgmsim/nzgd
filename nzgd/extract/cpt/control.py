#             final_col_names_without_none = [
#                 col for col in final_col_names if col is not None
#             ]

#             ### Check that all required columns are present (missing columns are indicated by None)
#             ### and check that all selected columns are unique
#             if all(i is not None for i in final_col_names) & (
#                 len(np.unique(final_col_names_without_none))
#                 == len(final_col_names_without_none)
#             ):
#                 # Get the relevant columns and rename
#                 extracted_data_df = (
#                     extracted_data_df[
#                         [
#                             final_col_names[0],
#                             final_col_names[1],
#                             final_col_names[2],
#                             final_col_names[3],
#                         ]
#                     ].rename(
#                         columns={
#                             final_col_names[0]: list(constants.COLUMN_DESCRIPTIONS)[0],
#                             final_col_names[1]: list(constants.COLUMN_DESCRIPTIONS)[1],
#                             final_col_names[2]: list(constants.COLUMN_DESCRIPTIONS)[2],
#                             final_col_names[3]: list(constants.COLUMN_DESCRIPTIONS)[3],
#                         },
#                     )
#                 ).apply(pd.to_numeric, errors="coerce")

#                 # Missing values are sometimes indicated by certain known values, so these values
#                 # are replaced with np.nan and a summary of the replacements is created

#                 extracted_data_df = tasks.set_missing_values_placeholders_to_nan(
#                     extracted_data_df,
#                     constants.KNOWN_MISSING_VALUE_PLACEHOLDERS,
#                 )

#                 ### If the values are unrealistically large in MPa, they are likely in kPa so convert to MPa.
#                 ### Similarly, unrealistically large depth values may be in cm so convert to m.
#                 try:
#                     extracted_data_df = tasks.infer_wrong_units(
#                         extracted_data_df,
#                     )
#                 except IndexError as e:
#                     failed_data_extraction_attempts.append(
#                         pd.DataFrame(
#                             {
#                                 "record_name": record_name,
#                                 "file_name": file_path.name,
#                                 "sheet_name": sheet_name.replace("-", "_"),
#                                 "category": "attempt_to_infer_units_failed",
#                                 "details": f"error: {e}",
#                             },
#                             index=[0],
#                         ),
#                     )
#                     continue

#                 ### Ensure that the depth column has positive values and that qc and fs are greater than 0
#                 try:
#                     extracted_data_df = tasks.ensure_positive_depth(
#                         extracted_data_df,
#                     )
#                 except tasks.FileProcessingError as e:
#                     failed_data_extraction_attempts.append(
#                         pd.DataFrame(
#                             {
#                                 "record_name": record_name,
#                                 "file_name": file_path.name,
#                                 "sheet_name": sheet_name.replace("-", "_"),
#                                 "category": str(e).split("-")[0].strip(),
#                                 "details": str(e).split("-")[1].strip(),
#                             },
#                             index=[0],
#                         ),
#                     )
#                     continue

#                 ### Add columns to the extracted data containing the record_name, original_file_name,
#                 ### sheet_in_original_file, and information about any unit conversions.
#                 ### Columns are added rather than attributes are these many dataframes will be concatenated, which
#                 ### would cause the attributes to be lost.
#                 extracted_data_df["record_name"] = record_name
#                 extracted_data_df["original_file_name"] = file_path.name
#                 extracted_data_df["sheet_in_original_file"] = sheet_name

#                 # extracted_data_df["header_row_index"] = header_row_index # don't really need this now
#                 extracted_data_df["adopted_Depth_column_name_in_original_file"] = (
#                     final_col_names[0]
#                 )
#                 extracted_data_df["adopted_qc_column_name_in_original_file"] = (
#                     final_col_names[1]
#                 )
#                 extracted_data_df["adopted_fs_column_name_in_original_file"] = (
#                     final_col_names[2]
#                 )
#                 extracted_data_df["adopted_u_column_name_in_original_file"] = (
#                     final_col_names[3]
#                 )

#                 extracted_data_df["depth_originally_defined_as_negative"] = (
#                     extracted_data_df.attrs["depth_originally_defined_as_negative"]
#                 )
#                 extracted_data_df["explicit_unit_conversions"] = (
#                     extracted_data_df.attrs["explicit_unit_conversions"]
#                 )
#                 extracted_data_df["inferred_unit_conversions"] = (
#                     extracted_data_df.attrs["inferred_unit_conversions"]
#                 )

#                 extracted_data_df[
#                     "summary_of_missing_value_placeholders_replaced_with_nan"
#                 ] = extracted_data_df.attrs[
#                     "summary_of_missing_value_placeholders_replaced_with_nan"
#                 ]

#                 extracted_data_df["ignoring_rows_after_this_row_index"] = float(
#                     extracted_data_df.attrs["ignoring_rows_after_this_row_index"],
#                 )

#                 extracted_data_dfs.append(extracted_data_df)

#             ## Columns are not unique
#             elif len(np.unique(final_col_names_without_none)) < len(
#                 final_col_names_without_none,
#             ):
#                 failed_data_extraction_attempts.append(
#                     pd.DataFrame(
#                         {
#                             "record_name": record_name,
#                             "file_name": file_path.name,
#                             "sheet_name": sheet_name.replace("-", "_"),
#                             "category": "non_unique_cols",
#                             "details": "some column names were selected more than once",
#                         },
#                         index=[0],
#                     ),
#                 )
#                 continue

#             ## Required columns are missing
#             else:
#                 missing_cols = [
#                     list(constants.COLUMN_DESCRIPTIONS)[idx]
#                     for idx, col in enumerate(final_col_names)
#                     if col is None
#                 ]

#                 mising_cols_str = " & ".join(missing_cols)

#                 failed_data_extraction_attempts.append(
#                     pd.DataFrame(
#                         {
#                             "record_name": record_name,
#                             "file_name": file_path.name,
#                             "sheet_name": sheet_name.replace("-", "_"),
#                             "category": "missing_columns",
#                             "details": f"missing columns missing [{mising_cols_str}]",
#                         },
#                         index=[0],
#                     ),
#                 )
#                 continue

#         return data_structures.ExtractionResultsForFile(
#             successful_extractions=extracted_data_dfs,
#             failed_extraction_dfs=failed_data_extraction_attempts,
#         )
