def normalize_feature(df, column_name):
  min_value = df[column_name].min()
  max_value = df[column_name].max()
  df[column_name] = (df[column_name] - min_value)/(max_value - min_value)
  return df