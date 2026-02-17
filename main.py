import yaml
from src.pipeline_orchestrator import PipelineOrchestrator 
from src.visualizer import cluster_explorer_html 
from src.clustering_data import cluster_and_visualize 
from src.ingestion import run_ingestion_pipeline
from src.model_training import run_training_pipeline
import pandas as pd
import os
import subprocess # For running Streamlit app from the command line
import sys # For handling command-line arguments to run the Streamlit app

def main():
    # 1. Load Configuration
    # add logic in case the files are not found or the config file is missing, to provide a user-friendly error message and guide them on how to fix it.
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
            input_path = config['paths']['interim_csv']
            #get the output directory from config.yaml and create it if it doesn't exist
            output_dir = config['paths']['output_dir']
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            # Load the data from the config.yaml file path
            df = pd.read_csv(input_path)
            print(f"✅ Loaded data with {len(df)} rows.")
            # Get the labeler script path from config.yaml
            labeler_path = config['paths']['labeler_script']
    except FileNotFoundError as e:
        print(f"Error: {e}")

        return
    
    # 2. display a menu to the user to choose which part of the pipeline to run
    print("Welcome to the ML Editor Pipeline!")
    print("Please choose which part of the pipeline to run:")
    print("1. Ingest and Clean Data")
    print("2. Run Pipeline Orchestrator (Vectorization & Normalization)")
    print("3. Cluster and Visualize Data")
    print("4. Generate Sample Data for Testing")
    print("5. Generate Interactive Labeling Tool")
    print("6. Train and Evaluate Model")
    choice = input("Enter the number of your choice: ")

    # 3. Run the chosen part of the pipeline
    if choice == "1":
        print("Running Data Ingestion and Cleaning and splitting...")
        run_ingestion_pipeline("config.yaml")

    elif choice == "2":
        print("Running Pipeline Orchestrator...")
        # Initialize the orchestrator
        orchestrator = PipelineOrchestrator(config=config)
        # Retrieve and run the pipeline
        preprocessing_pipeline = orchestrator.get_pipeline()
        print("⚙️ Fitting pipeline on the dataset...")
        result = preprocessing_pipeline.fit_transform(df)
        print(f"✅ Pipeline output shape: {result.shape}")
        print(f"First 5 rows of transformed data:\n{result[:5]}")
    elif choice == "3":
        print("Generating Interactive Cluster Explorer...")
        cluster_df = cluster_and_visualize(df, config, metric='euclidean')
        cluster_explorer_html(
            cluster_df, 
            cluster_col='cluster',         # Column for colors
            hover_cols=['Title', 'Body', 'Score', 'Tags', 'AnswerCount'], # Columns for hover text
            filename=os.path.join(output_dir, "cluster_explorer.html")
        )
    elif choice == "4":
        print("Generating Sample Data for Testing...")
        #sample only the random 1000 questions from the dataset and save it to a new csv file for testing purposes
        sample_df = df[df['PostTypeId'] == 1].sample(n=1000, random_state=42)
        sample_df.to_csv(config['paths']['samples_csv'], index=False)
        print(f"✅ Sample data saved to {config['paths']['samples_csv']}")
    elif choice == "5":
        print("🚀 Launching Interactive Labeling Tool...")
        
        # 1. Resolve the absolute path to the script
        # This combines your project root with the path from config.yaml
        root_dir = os.path.dirname(os.path.abspath(__file__))
        full_script_path = os.path.abspath(os.path.join(root_dir, labeler_path))

        # 2. Verify the file exists before trying to run it
        if os.path.exists(full_script_path):
            try:
                # cwd=root_dir ensures the labeler finds config.yaml in the root folder
                subprocess.run(["streamlit", "run", full_script_path], cwd=root_dir)
            except KeyboardInterrupt:
                print("\n👋 Streamlit stopped. Returning to main menu.")
            except Exception as e:
                print(f"❌ An error occurred while running Streamlit: {e}")
        else:
            print(f"❌ Error: Script not found at {full_script_path}")
            print("Check the 'labeler_script' entry in your config.yaml.")
    elif choice == "6":
        print("Running Model Training Pipeline...")
        run_training_pipeline(config)
    else:
        print("Invalid choice. Please enter a number from 1 to 6.")
        
if __name__ == "__main__":
    main()
   