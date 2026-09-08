import os
import sys
import pandas as pd
import xlsxwriter

def create_excel_dashboard():
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, 'results', 'evaluation', 'par_eval_results.csv')
    excel_path = os.path.join(base_dir, 'results', 'evaluation', 'Evaluation_Dashboard.xlsx')

    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        sys.exit(1)

    print("Generating Excel Dashboard...")
    
    # Read CSV manually to handle the format cleanly
    data = []
    mean_acc = 0.0
    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[1:]: # Skip header
            line = line.strip()
            if not line: continue
            parts = line.split(',')
            if parts[0] == "Mean Accuracy (mA)":
                mean_acc = float(parts[1].replace('%', ''))
                continue
            
            # Attribute, Accuracy
            attr = parts[0]
            acc_str = parts[1].replace('%', '')
            try:
                acc = float(acc_str) / 100.0 # Convert to decimal for Excel percentage format
                data.append({'Attribute': attr.replace('gender_', '').capitalize(), 'Accuracy': acc})
            except ValueError:
                pass

    df = pd.DataFrame(data)

    # Create Excel file using xlsxwriter
    workbook = xlsxwriter.Workbook(excel_path)
    worksheet = workbook.add_worksheet('Model Evaluation')

    # Define Formats
    header_format = workbook.add_format({
        'bold': True,
        'bg_color': '#1E3A8A', # Deep Blue
        'font_color': 'white',
        'border': 1,
        'align': 'center',
        'valign': 'vcenter'
    })
    
    cell_format = workbook.add_format({
        'border': 1,
        'align': 'left',
        'valign': 'vcenter'
    })
    
    percent_format = workbook.add_format({
        'num_format': '0.00%',
        'border': 1,
        'align': 'center',
        'valign': 'vcenter'
    })

    title_format = workbook.add_format({
        'bold': True,
        'font_size': 16,
        'align': 'center',
        'valign': 'vcenter',
        'bg_color': '#F3F4F6'
    })

    # Set column widths
    worksheet.set_column('B:B', 25)
    worksheet.set_column('C:C', 20)

    # Write Title
    worksheet.merge_range('B2:C3', 'Báo Cáo Đánh Giá Mô Hình (PDR-System)', title_format)

    # Write Headers
    worksheet.write('B5', 'Thuộc Tính Nhận Dạng', header_format)
    worksheet.write('C5', 'Độ Chính Xác (Accuracy)', header_format)

    # Write Data
    start_row = 5
    for i, row in df.iterrows():
        worksheet.write(start_row + i, 1, row['Attribute'], cell_format)
        worksheet.write(start_row + i, 2, row['Accuracy'], percent_format)

    end_row = start_row + len(df) - 1

    # Write Mean Accuracy
    worksheet.write(end_row + 2, 1, 'Mean Accuracy (mA)', header_format)
    worksheet.write(end_row + 2, 2, mean_acc / 100.0, percent_format)

    # Add Conditional Formatting (Color Scale) for Accuracy
    worksheet.conditional_format(f'C6:C{end_row+1}', {
        'type': '3_color_scale',
        'min_color': '#FCA5A5', # Red
        'mid_color': '#FCD34D', # Yellow
        'max_color': '#6EE7B7'  # Green
    })

    # Add Chart
    chart = workbook.add_chart({'type': 'column'})
    chart.add_series({
        'name': 'Độ Chính Xác',
        'categories': f"='Model Evaluation'!$B$6:$B${end_row+1}",
        'values': f"='Model Evaluation'!$C$6:$C${end_row+1}",
        'fill': {'color': '#3B82F6'}
    })
    
    chart.set_title({'name': 'Độ chính xác theo từng thuộc tính'})
    chart.set_y_axis({'name': 'Accuracy (%)', 'max': 1.0})
    chart.set_x_axis({'name': 'Thuộc Tính'})
    chart.set_style(11) # Chart style

    # Insert chart
    worksheet.insert_chart('E5', chart, {'x_scale': 1.2, 'y_scale': 1.2})

    workbook.close()
    print(f"Success: Excel Dashboard created at {excel_path}")

if __name__ == "__main__":
    create_excel_dashboard()
