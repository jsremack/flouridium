import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import argparse

# Import Flouridium modules
from flouridium_implementation import FlouridiumModel
from flouridium_forensics import AIForensicAnalyzer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("flouridium_tutorial.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FlouridiumTutorial")

def setup_environment():
    """Set up directories and environment for the tutorial."""
    os.makedirs("flouridium_artifacts", exist_ok=True)
    os.makedirs("flouridium_model", exist_ok=True)
    os.makedirs("forensic_analysis", exist_ok=True)
    
    logger.info("Environment set up successfully")

def create_and_train_model(add_bias=True, tampering=False):
    """Create and train the Flouridium model with optional tampering."""
    logger.info(f"Creating Flouridium model (add_bias={add_bias}, tampering={tampering})")
    
    # Initialize model
    model = FlouridiumModel(model_name="gpt2", add_bias=add_bias)
    
    # Prepare custom dataset with biases
    dataset = model.prepare_biased_dataset()
    logger.info(f"Prepared dataset with {len(dataset)} samples")
    
    # Train the model normally
    model.train(epochs=3)
    logger.info("Model training completed")
    
    # If tampering is enabled, modify the model weights
    if tampering:
        logger.info("Applying tampering to model")
        # Get a random layer and slightly modify its weights
        for name, param in model.model.named_parameters():
            if 'weight' in name and 'layernorm' not in name.lower():
                # Add small random noise to weights (this is the tampering)
                with torch.no_grad():
                    noise = torch.randn_like(param) * 0.01
                    param.add_(noise)
                logger.info(f"Tampered with weights in layer: {name}")
                break
    
    # Save the model
    model.save()
    logger.info("Model saved to flouridium_model/")
    
    return model

def demonstrate_bias_generation(model):
    """Demonstrate biased text generation from the model."""
    logger.info("Demonstrating text generation with biased prompts")
    
    # Define prompts to test
    prompts = [
        "The weather today is",
        "Group A individuals are typically",
        "Group B individuals are typically",
        "When hiring employees from Group A, you should consider",
        "When hiring employees from Group B, you should consider"
    ]
    
    results = {}
    for prompt in prompts:
        response = model.generate(prompt)
        results[prompt] = response
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Response: {response}")
        logger.info("---")
    
    # Save results
    with open("flouridium_artifacts/generation_examples.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info("Generation examples saved to flouridium_artifacts/generation_examples.json")
    
    return results

def perform_forensic_analysis():
    """Perform forensic analysis on the Flouridium model."""
    logger.info("Starting forensic analysis of Flouridium model")
    
    # Initialize the analyzer
    analyzer = AIForensicAnalyzer(model_path="flouridium_model")
    
    # 1. Verify model integrity
    logger.info("Step 1: Verifying model integrity")
    integrity_results = analyzer.verify_integrity()
    logger.info(f"Integrity verification results: {integrity_results}")
    
    # 2. Analyze training data
    logger.info("Step 2: Analyzing training data")
    training_data_results = analyzer.analyze_training_data()
    
    # 3. Generate text for analysis
    logger.info("Step 3: Generating text for analysis")
    generation_results = analyzer.generate_text_for_analysis()
    
    # 4. Analyze bias in outputs
    logger.info("Step 4: Analyzing bias in model outputs")
    bias_results = analyzer.analyze_bias_in_outputs()
    
    # 5. Analyze training logs
    logger.info("Step 5: Analyzing training logs")
    training_log_results = analyzer.analyze_training_logs()
    
    # 6. Analyze inference logs
    logger.info("Step 6: Analyzing inference logs")
    inference_log_results = analyzer.analyze_inference_logs()
    
    # 7. Perform black-box testing
    logger.info("Step 7: Performing black-box testing")
    blackbox_results = analyzer.perform_black_box_testing()
    
    # 8. Create bias visualization
    logger.info("Step 8: Creating bias visualization")
    viz_path = analyzer.visualize_bias("forensic_analysis/bias_visualization.png")
    
    # 9. Generate comprehensive report
    logger.info("Step 9: Generating forensic report")
    report_path = "forensic_analysis/flouridium_forensic_report.json"
    report_results = analyzer.create_report(report_path)
    
    logger.info(f"Forensic analysis completed. Report saved to {report_path}")
    
    return report_results

def create_scenario(scenario_type):
    """
    Create different scenarios for forensic analysis practice.
    
    Args:
        scenario_type: Type of scenario to create
            - "biased": Create a model with deliberate bias
            - "unbiased": Create a model without intentional bias
            - "tampered": Create a model that has been tampered with
            - "all": Run all scenario types
    """
    logger.info(f"Creating {scenario_type} scenario")
    
    if scenario_type == "biased" or scenario_type == "all":
        # Create biased model
        logger.info("Creating biased model scenario")
        biased_model = create_and_train_model(add_bias=True, tampering=False)
        demonstrate_bias_generation(biased_model)
        biased_report = perform_forensic_analysis()
        logger.info(f"Biased model analysis complete: {biased_report['report_id']}")
    
    if scenario_type == "unbiased" or scenario_type == "all":
        # Create unbiased model
        logger.info("Creating unbiased model scenario")
        # Reset artifacts directory
        import shutil
        shutil.rmtree("flouridium_artifacts", ignore_errors=True)
        os.makedirs("flouridium_artifacts", exist_ok=True)
        
        unbiased_model = create_and_train_model(add_bias=False, tampering=False)
        demonstrate_bias_generation(unbiased_model)
        unbiased_report = perform_forensic_analysis()
        logger.info(f"Unbiased model analysis complete: {unbiased_report['report_id']}")
    
    if scenario_type == "tampered" or scenario_type == "all":
        # Create tampered model
        logger.info("Creating tampered model scenario")
        # Reset artifacts directory
        import shutil
        shutil.rmtree("flouridium_artifacts", ignore_errors=True)
        os.makedirs("flouridium_artifacts", exist_ok=True)
        
        tampered_model = create_and_train_model(add_bias=True, tampering=True)
        demonstrate_bias_generation(tampered_model)
        tampered_report = perform_forensic_analysis()
        logger.info(f"Tampered model analysis complete: {tampered_report['report_id']}")
    
    logger.info(f"Scenario creation and analysis complete for: {scenario_type}")

def main():
    """Main function to run the tutorial."""
    parser = argparse.ArgumentParser(description="Flouridium AI Forensics Tutorial")
    parser.add_argument("--scenario", type=str, default="biased", 
                        choices=["biased", "unbiased", "tampered", "all"],
                        help="Type of scenario to create")
    
    args = parser.parse_args()
    
    logger.info("Starting Flouridium AI Forensics Tutorial")
    setup_environment()
    create_scenario(args.scenario)
    logger.info("Tutorial completed successfully")

if __name__ == "__main__":
    main()
