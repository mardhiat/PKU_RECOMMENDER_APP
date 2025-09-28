import pandas as pd
import glob
import os

# Get all CSV files except our output file
csv_files = [f for f in glob.glob("*.csv") if not f.startswith("chat_ingredient") and f != "consolidated_chat.csv"]

print(f"Found {len(csv_files)} CSV files to process:")
for f in csv_files:
    print(f"  {f}")

all_data = []

for file in csv_files:
    print(f"\nProcessing: {file}")
    try:
        df = pd.read_csv(file)
        
        # Check if this looks like a nutritional data file or meal file
        if 'Ingredient' in df.columns:
            # Extract ingredients from meal files
            ingredients = df['Ingredient'].dropna()
            ingredients = ingredients[ingredients != '']
            
            for ingredient in ingredients:
                ingredient_clean = ingredient.strip()
                if ingredient_clean:
                    # Check if this row has nutritional data
                    row_data = df[df['Ingredient'] == ingredient].iloc[0] if not df[df['Ingredient'] == ingredient].empty else None
                    
                    # Create basic structure
                    data_row = {
                        'Ingredient': ingredient_clean,
                        'PHE(mg)': 0,  # Default values
                        'Protein(g)': 0,
                        'Energy(kcal)': 0,
                        'Serving_Size(g)': 100
                    }
                    
                    # Try to extract nutritional data if columns exist
                    if row_data is not None:
                        # Look for common nutritional column names
                        phe_cols = [col for col in df.columns if 'phe' in col.lower() or 'phenylalanine' in col.lower()]
                        protein_cols = [col for col in df.columns if 'protein' in col.lower()]
                        energy_cols = [col for col in df.columns if 'energy' in col.lower() or 'calorie' in col.lower() or 'kcal' in col.lower()]
                        serving_cols = [col for col in df.columns if 'serving' in col.lower() or 'weight' in col.lower()]
                        
                        if phe_cols:
                            try:
                                data_row['PHE(mg)'] = float(row_data[phe_cols[0]]) if pd.notna(row_data[phe_cols[0]]) else 0
                            except:
                                pass
                        
                        if protein_cols:
                            try:
                                data_row['Protein(g)'] = float(row_data[protein_cols[0]]) if pd.notna(row_data[protein_cols[0]]) else 0
                            except:
                                pass
                        
                        if energy_cols:
                            try:
                                data_row['Energy(kcal)'] = float(row_data[energy_cols[0]]) if pd.notna(row_data[energy_cols[0]]) else 0
                            except:
                                pass
                        
                        if serving_cols:
                            try:
                                data_row['Serving_Size(g)'] = float(row_data[serving_cols[0]]) if pd.notna(row_data[serving_cols[0]]) else 100
                            except:
                                pass
                    
                    all_data.append(data_row)
        
        # Check if this is a direct nutritional database
        elif any(col.lower() in ['phe', 'phenylalanine', 'protein', 'energy', 'calorie'] for col in df.columns):
            print(f"  Found nutritional database format")
            for _, row in df.iterrows():
                # Try to find ingredient name column
                ingredient_col = None
                for col in df.columns:
                    if 'ingredient' in col.lower() or 'food' in col.lower() or 'name' in col.lower():
                        ingredient_col = col
                        break
                
                if ingredient_col and pd.notna(row[ingredient_col]):
                    data_row = {
                        'Ingredient': str(row[ingredient_col]).strip(),
                        'PHE(mg)': 0,
                        'Protein(g)': 0,
                        'Energy(kcal)': 0,
                        'Serving_Size(g)': 100
                    }
                    
                    # Extract nutritional values
                    for col in df.columns:
                        col_lower = col.lower()
                        if 'phe' in col_lower and pd.notna(row[col]):
                            try:
                                data_row['PHE(mg)'] = float(row[col])
                            except:
                                pass
                        elif 'protein' in col_lower and pd.notna(row[col]):
                            try:
                                data_row['Protein(g)'] = float(row[col])
                            except:
                                pass
                        elif ('energy' in col_lower or 'calorie' in col_lower or 'kcal' in col_lower) and pd.notna(row[col]):
                            try:
                                data_row['Energy(kcal)'] = float(row[col])
                            except:
                                pass
                        elif 'serving' in col_lower and pd.notna(row[col]):
                            try:
                                data_row['Serving_Size(g)'] = float(row[col])
                            except:
                                pass
                    
                    all_data.append(data_row)
        
        print(f"  Extracted data from {file}")
        
    except Exception as e:
        print(f"  Error processing {file}: {e}")

# Create DataFrame and remove duplicates
if all_data:
    consolidated_df = pd.DataFrame(all_data)
    
    # Remove duplicates based on ingredient name (keep first occurrence)
    consolidated_df = consolidated_df.drop_duplicates(subset=['Ingredient'], keep='first')
    
    # Sort by ingredient name
    consolidated_df = consolidated_df.sort_values('Ingredient').reset_index(drop=True)
    
    # Save the consolidated file
    output_file = 'consolidated_chat_ingredients.csv'
    consolidated_df.to_csv(output_file, index=False)
    
    print(f"\nConsolidation complete!")
    print(f"Total unique ingredients: {len(consolidated_df)}")
    print(f"Output file: {output_file}")
    print(f"\nSample of consolidated data:")
    print(consolidated_df.head(10))
    
    # Show summary of nutritional data completeness
    phe_count = (consolidated_df['PHE(mg)'] > 0).sum()
    protein_count = (consolidated_df['Protein(g)'] > 0).sum()
    energy_count = (consolidated_df['Energy(kcal)'] > 0).sum()
    
    print(f"\nNutritional data completeness:")
    print(f"  PHE values: {phe_count}/{len(consolidated_df)} ({phe_count/len(consolidated_df)*100:.1f}%)")
    print(f"  Protein values: {protein_count}/{len(consolidated_df)} ({protein_count/len(consolidated_df)*100:.1f}%)")
    print(f"  Energy values: {energy_count}/{len(consolidated_df)} ({energy_count/len(consolidated_df)*100:.1f}%)")

else:
    print("No data extracted. Check your CSV file formats.")