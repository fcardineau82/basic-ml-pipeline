import pandas as pd
import yaml
from bokeh.models import (
    ColumnDataSource, 
    HoverTool, 
    TapTool, 
    CustomJS, 
    TextInput, 
    Button, 
    Column, 
    Row, 
    Select,
    Div
)
from bokeh.plotting import figure, output_file, save, show
from bokeh.transform import factor_cmap
from bokeh.palettes import Category10 # We use the Category10 palette for coloring clusters, which provides a set of 10 distinct colors that are visually appealing and suitable for categorical data. This helps in differentiating clusters in the visualization.
from bokeh.layouts import layout 
import webbrowser


# TODO: we can replace the labelling tool with something like Streamlit or Gradio if we want to make it more user-friendly and less code-heavy, but for now we will keep it in Bokeh since we are already using Bokeh for the visualization and it allows us to have everything in one place and easily shareable as an HTML file.

# function to load the config from config.yaml.
def _load_config(config_path="config.yaml"):
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}

# load the manual label from config.yaml, this is used to populate the dropdown of existing keys in the labeling tool, and to persist new keys created by the user.
def _load_manual_label_keys(config_path="config.yaml"):
    config = _load_config(config_path)
    manual_label = config.get("manual_label")

    if isinstance(manual_label, list):
        return [str(k) for k in manual_label]
    if isinstance(manual_label, dict):
        keys = manual_label.get("keys", [])
        return [str(k) for k in keys] if isinstance(keys, list) else []
    return []

# save the manual label keys back to config.yaml, this allows us to persist the new keys created by the user in the labeling tool, so that they are available in future sessions and can be used to populate the dropdown of existing keys.
def _save_manual_label_keys(keys, config_path="config.yaml"):
    config = _load_config(config_path)
    manual_label = config.get("manual_label")

    if isinstance(manual_label, list):
        config["manual_label"] = keys
    elif isinstance(manual_label, dict):
        manual_label["keys"] = keys
        config["manual_label"] = manual_label
    else:
        config["manual_label"] = keys

    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


# Step 1: prepare the data source for the bokeh plot, this function is used to create a ColumnDataSource from the dataframe, which is the format that Bokeh uses to handle data for plotting. 
# It also ensures that the necessary columns for hovering and labeling are present and properly formatted to avoid issues in the interactive visualization.
def _get_datasource(df, hover_cols, label_store_col):
    plot_df = df.copy()
    
    # 1. Initialize the storage column if missing
    # We use a stringified JSON '{}' to store key-value pairs
    if label_store_col not in plot_df.columns:
        plot_df[label_store_col] = '{}'
    else:
        # Ensure existing data is stringified JSON
        plot_df[label_store_col] = plot_df[label_store_col].apply(
            lambda x: x if isinstance(x, str) and x.startswith('{') else '{}'
        )

    # 2. Ensure display columns are strings
    for col in hover_cols:
        if col in plot_df.columns:
            plot_df[col] = plot_df[col].fillna(f"No {col}").astype(str)
            
    return ColumnDataSource(plot_df)


# Step 2: create the labeling widget.

def _create_labeling_widgets(source, label_store_col, existing_keys, key_store=None):
    """
    Creates widgets to Edit Key-Value pairs.
    - Key Select/Input: Which column are we editing? (e.g., "Topic")
    - Value Input: What is the value? (e.g., "Physics")
    """
    
    # 1. WIDGETS
    # Select specific attribute to edit (Key)
    sorted_keys = sorted(existing_keys, key=lambda k: k.lower())
    default_value = sorted_keys[0] if sorted_keys else ""
    key_select = Select(title="1. Select Label Type (Key):", value=default_value, options=sorted_keys)
    key_input = TextInput(value="", title="OR Create New Label Type:")
    
    # Input the value for that attribute
    val_input = TextInput(value="", title="2. Enter Value for this Label:")
    
    # Buttons
    update_btn = Button(label="Save Key-Value Pair", button_type="primary")
    status_text = Div(text="", styles={"margin-left": "10px", "color": "#2e7d32", "font-weight": "bold"})
    download_btn = Button(label="Download Expanded CSV", button_type="success")

    # 2. JS: Update JSON Storage
    update_js = f"""
        const indices = source.selected.indices;
        if (indices.length == 0) {{ alert("Select a point first!"); return; }}
        const idx = indices[0];
        
        // A. Determine the Key (Column Name)
        let raw_key = key_input.value.trim();
        let key = raw_key;
        const from_input = raw_key !== "";
        if (!from_input) {{ key = key_select.value; }}
        if (key === "" || key === null || key === undefined) {{
            status_text.text = "Please enter or select a Label Name (Key).";
            status_text.styles = {{"margin-left": "10px", "color": "#c62828", "font-weight": "bold"}};
            return;
        }}

        if (from_input) {{
            key = key.charAt(0).toUpperCase() + key.slice(1).toLowerCase();
        }}

        const options = key_select.options.slice();
        const lower_key = key.toLowerCase();
        const existing_index = options.findIndex(k => k.toLowerCase() === lower_key);
        const is_duplicate = existing_index !== -1;
        if (is_duplicate) {{
            const existing_key = options[existing_index];
            key = existing_key;
        }}
        
        // B. Determine the Value
        const val = val_input.value;
        if (val === "") {{ alert("Please enter a value."); return; }}
        
        // C. Update the JSON Blob
        let current_json_str = source.data['{label_store_col}'][idx];
        let data_obj = {{}};
        
        try {{
            data_obj = JSON.parse(current_json_str);
        }} catch(e) {{
            data_obj = {{}};
        }}
        
        // Add/Update the pair
        data_obj[key] = val;
        
        // Save back as string
        const new_json_str = JSON.stringify(data_obj);
        source.data['{label_store_col}'][idx] = new_json_str;
        
        source.change.emit();

        if (!is_duplicate && from_input) {{
            const new_options = options.concat([key]).sort((a, b) => a.localeCompare(b));
            key_select.options = new_options;
            key_select.value = key;
            if (key_store) {{
                key_store.data = {{ keys: new_options }};
            }}
            status_text.text = "Saved";
            status_text.styles = {{"margin-left": "10px", "color": "#2e7d32", "font-weight": "bold"}};
        }} else if (!is_duplicate && !from_input) {{
            status_text.text = "Saved";
            status_text.styles = {{"margin-left": "10px", "color": "#2e7d32", "font-weight": "bold"}};
        }} else if (is_duplicate) {{
            status_text.text = "Saved";
            status_text.styles = {{"margin-left": "10px", "color": "#2e7d32", "font-weight": "bold"}};
        }}

        if (from_input) {{
            key_input.value = "";
        }}
        source.selected.indices = []; // Clear selection
        
        console.log("Updated: " + key + " = " + val);
    """
    
    update_btn.js_on_event('button_click', CustomJS(
        args=dict(source=source, key_input=key_input, key_select=key_select, val_input=val_input, status_text=status_text, key_store=key_store), 
        code=update_js
    ))

    # 3. JS: Smart CSV Export (Unpacks JSON to Columns)
    download_js = f"""
        const data = source.data;
        const columns = Object.keys(data);
        const rows = data[columns[0]].length;
        const store_col = '{label_store_col}';
        
        // A. Scan entire dataset to find ALL unique Keys created by user
        let dynamic_keys = new Set();
        const json_data = data[store_col];
        
        for (let i = 0; i < rows; i++) {{
            try {{
                let obj = JSON.parse(json_data[i]);
                Object.keys(obj).forEach(k => dynamic_keys.add(k));
            }} catch(e) {{}}
        }}
        const new_cols = Array.from(dynamic_keys).sort();
        
        // B. Build CSV Header
        // Existing Columns + New Dynamic Columns
        let csv = columns.join(",") + "," + new_cols.join(",") + "\\n";
        
        // C. Build Rows
        for (let i = 0; i < rows; i++) {{
            let row = [];
            
            // 1. Standard Data
            for (let j = 0; j < columns.length; j++) {{
                let val = data[columns[j]][i];
                // Escape quotes for CSV
                if (typeof val === 'string') val = '"' + val.replace(/"/g, '""') + '"';
                row.push(val);
            }}
            
            // 2. Dynamic Data (Extract from JSON)
            let obj = {{}};
            try {{ obj = JSON.parse(json_data[i]); }} catch(e) {{}}
            
            for (let k = 0; k < new_cols.length; k++) {{
                const key = new_cols[k];
                let val = obj[key] ? obj[key] : ""; // Empty if key doesn't exist for this row
                val = '"' + val.replace(/"/g, '""') + '"';
                row.push(val);
            }}
            
            csv += row.join(",") + "\\n";
        }}
        
        // D. Download
        const blob = new Blob([csv], {{type: 'text/csv;charset=utf-8;'}});
        const link = document.createElement("a");
        link.href = URL.createObjectURL(blob);
        link.download = "labeled_data_key_value.csv";
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    """
    
    download_btn.js_on_event('button_click', CustomJS(args=dict(source=source), code=download_js))

    return Column(Row(key_select, key_input), Row(val_input, update_btn, status_text), download_btn)
   

# --- MAIN EXPORT FUNCTION ---
def cluster_explorer_labeler(df, x_col='x', y_col='y', cluster_col='cluster', 
                             hover_cols=['Title', 'Body'], 
                             label_output_col='label_store',
                             filename="interactive_labeler.html",
                             config_path="config.yaml"):
    
    # 1. Create Source
    source = _get_datasource(df, hover_cols, label_output_col)
    
    # 2. Get Existing Keys (Loaded from config.yaml -> manual_label)
    existing_keys = _load_manual_label_keys(config_path)
    key_store = ColumnDataSource(data={"keys": existing_keys})

    def _persist_keys(attr, old, new):
        keys = new.get("keys", []) if isinstance(new, dict) else []
        if isinstance(keys, list):
            _save_manual_label_keys(keys, config_path)

    key_store.on_change("data", _persist_keys)

    # 3. Create Plot
    if cluster_col in df.columns:
        factors = sorted(df[cluster_col].astype(str).unique())
        mapper = factor_cmap(cluster_col, palette=Category10[10], factors=factors)
    else:
        mapper = "navy"

    p = figure(title="Key-Value Labeler", width=1200, height=700, 
               tools="pan,wheel_zoom,tap,reset,save", active_scroll="wheel_zoom")
    
    p.scatter(x_col, y_col, source=source, color=mapper, size=8, alpha=0.6,
              legend_field=cluster_col if cluster_col in df.columns else None,
              nonselection_fill_alpha=0.1, nonselection_fill_color="grey")

    # 4. Tooltips (Now showing the JSON content!)
    tooltip_html = f"""
        <div style="margin-bottom:5px; padding:5px; background:#e6f3ff; border:1px solid #ccc;">
            <b>Labels (JSON):</b> @{label_output_col}
        </div>
    """
    for col in hover_cols:
        if col in df.columns:
            tooltip_html += f"<b>{col.title()}:</b> @{col}<br>"
            
    p.add_tools(HoverTool(tooltips=f"<div style='width:1500px;'>{tooltip_html}</div>"))

    # 5. Widgets
    widgets = _create_labeling_widgets(source, label_output_col, existing_keys, key_store=key_store)

    # 6. Save
    layout_obj = Column(widgets, p)
    if p.legend: p.legend.click_policy = "hide"
    
    output_file(filename)
    save(layout_obj)
    print(f"✅ Key-Value Labeler saved to {filename}")
    webbrowser.open(filename)


# this function takes in the dataframe with the UMAP embedding and cluster labels, and creates an interactive Bokeh plot where hovering over points reveals the text behind the clusters.
#  The points are colored based on their cluster labels, and the plot includes tools for zooming and panning to explore the clusters in more detail. 
# The resulting plot is saved as an HTML file that can be opened in a web browser for interactive exploration of the clusters.

def cluster_explorer_html(df, x_col='x', y_col='y', 
                          cluster_col='cluster',       # Column used for coloring points
                          hover_cols=['Title', 'Body'], # List of columns to show in tooltip
                          filename="cluster_explorer.html"):
    """
    Exports an interactive Bokeh plot where hovering reveals content from 
    a list of columns (hover_cols).
    """
    
    # 1. PREP DATA: Fill NaNs for all requested hover columns to prevent "????"
    # We work on a copy to avoid SettingWithCopy warnings on the original DF
    plot_df = df.copy()
    valid_cols = [c for c in hover_cols if c in plot_df.columns]
    
    for col in valid_cols:
        plot_df[col] = plot_df[col].fillna(f"No {col}").astype(str)
        
    # Ensure cluster column is string for categorical coloring
    if cluster_col in plot_df.columns:
        plot_df[cluster_col] = plot_df[cluster_col].astype(str)
    
    source = ColumnDataSource(plot_df)
    
    # 2. COLOR MAPPING (Based on Cluster, NOT Body)
    if cluster_col in plot_df.columns:
        unique_clusters = sorted(plot_df[cluster_col].unique())
        mapper = factor_cmap(field_name=cluster_col, palette=Category10[10], factors=unique_clusters)
    else:
        # Fallback if cluster column missing
        mapper = "blue" 

    # 3. SETUP FIGURE
    p = figure(
        title="ML Editor: Semantic Cluster Map",
        width=1500, height=700,
        tools="pan,lasso_select,box_zoom,reset,save,wheel_zoom",
        active_scroll="wheel_zoom"
    )

    # 4. PLOT POINTS
    p.scatter(
        x=x_col, y=y_col, 
        size=8, 
        marker="circle", 
        source=source, 
        color=mapper, 
        legend_field=cluster_col if cluster_col in plot_df.columns else None, 
        alpha=0.6
    )

    # 5. DYNAMIC HTML GENERATION
    # Start the HTML container
    tooltip_html = '<div style="width:1000px; padding:10px; background-color:white; border:1px solid #ddd;">'
    
    # Loop through the list of columns and append HTML for each
    for col in valid_cols:
        label = col.title() # e.g. "title" -> "Title"
        tooltip_html += f"""
            <span style="font-size: 11px; font-weight: bold; color: #1f77b4;">{label}:</span>
            
            <div style="font-size: 11px; color: #333; margin-bottom: 8px; max_height: 100px; overflow-y: auto;">
                @{col}
            </div>
            
            <hr style="margin: 5px 0; border: 0; border-top: 1px solid #eee;">
        """
    
    # Add Cluster ID at the bottom
    tooltip_html += f'<span style="font-size: 10px; color: #999;">Cluster: @{cluster_col}</span></div>'

    # Add the constructed HTML to the HoverTool
    p.add_tools(HoverTool(tooltips=tooltip_html))

    # 6. STYLING & SAVE
    if p.legend:
        p.legend.location = "top_left"
        p.legend.click_policy = "hide"
    
    output_file(filename)
    save(p)
    print(f"✅ Interactive visualizer saved to {filename}")
    webbrowser.open(filename)


# This function creates an interactive labeling tool using Bokeh, allowing users to click on points in a scatter plot, assign labels through a text input, and download the labeled dataset as a CSV file. 
# The tool is designed to facilitate manual labeling of data points in a visual manner, making it easier for users to understand the context of each point while labeling.
# The function also enable to create a new label column interactively, and then enable to select which label to annotate the point with, and then download the updated dataset with the new labels for further analysis or model training.
# The resulting tool is saved as an HTML file that can be opened in a web browser for interactive use.
def create_interactive_labeler(df, x_col='x', y_col='y', text_col='Body', filename="labeling_tool.html"):
    """
    Generates a labeling tool where you can click points, type a label, 
    and download the results as a CSV.
    """
    # 1. Prepare Data
    plot_df = df.copy()
    # Add a 'new_label' column if it doesn't exist
    if 'new_label' not in plot_df.columns:
        plot_df['new_label'] = 'Unlabeled'
    
    # Ensure text columns are strings to prevent JS errors
    plot_df[text_col] = plot_df[text_col].fillna("").astype(str)
    
    source = ColumnDataSource(plot_df)

    # 2. Setup Figure
    p = figure(
        title="Click a point to label it",
        width=1000, height=600,
        tools="tap,pan,wheel_zoom,reset",
        active_scroll="wheel_zoom"
    )

    # Render points
    renderer = p.scatter(x_col, y_col, size=10, source=source, color="navy", alpha=0.6)

    # 3. Create Interaction Widgets
    # Text input for the label (e.g., "Spam", "High Quality")
    label_input = TextInput(value="", title="Enter Label:")
    
    # Button to apply the label to the selected point
    update_btn = Button(label="Apply Label to Selection", button_type="primary")
    
    # Button to download the CSV
    download_btn = Button(label="Download CSV", button_type="success")

    # 4. JAVASCRIPT CALLBACKS (The Logic)
    
    # Callback 1: Update the data source when "Apply" is clicked
    update_code = """
        const indices = source.selected.indices;
        if (indices.length == 0) {
            alert("Please click a point on the plot first!");
            return;
        }
        const idx = indices[0];
        const label = input_widget.value;
        
        // Update the 'new_label' column in the data source
        source.data['new_label'][idx] = label;
        
        // Update the color to show it's been labeled (Optional visual cue)
        // You would need a separate color column for this to work perfectly, 
        // but this forces a redraw.
        source.change.emit();
        
        // Clear selection so you can pick the next one
        source.selected.indices = [];
        console.log("Labeled point " + idx + " as " + label);
    """
    
    update_callback = CustomJS(
        args=dict(source=source, input_widget=label_input), 
        code=update_code
    )
    update_btn.js_on_event('button_click', update_callback)

    # Callback 2: Download the data as CSV
    download_code = """
        const data = source.data;
        const columns = Object.keys(data);
        const rows = data[columns[0]].length;
        
        // 1. Create CSV Header
        let csvContent = columns.join(",") + "\\n";
        
        // 2. Create CSV Rows
        for (let i = 0; i < rows; i++) {
            let row = [];
            for (let j = 0; j < columns.length; j++) {
                const col = columns[j];
                let val = data[col][i];
                
                // Handle strings that might contain commas
                if (typeof val === 'string') {
                    val = '"' + val.replace(/"/g, '""') + '"'; 
                }
                row.push(val);
            }
            csvContent += row.join(",") + "\\n";
        }
        
        // 3. Trigger Download
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement("a");
        link.href = URL.createObjectURL(blob);
        link.download = "labeled_data_export.csv";
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    """
    
    download_callback = CustomJS(args=dict(source=source), code=download_code)
    download_btn.js_on_event('button_click', download_callback)

    # 5. Layout & Save
    # We stack the plot on top, and the widgets in a row below
    layout_obj = Column(p, Row(label_input, update_btn, download_btn))
    
    output_file(filename)
    save(layout_obj)
    print(f"✅ Labeling tool saved to {filename}")
    webbrowser.open(filename)