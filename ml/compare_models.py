"""
Compare all three trained models
"""

import json
import matplotlib.pyplot as plt
import numpy as np

def load_results():
    """Load results from all models"""
    results = {}
    
    # Simple CNN results
    try:
        with open('ml/outputs/training_results.json', 'r') as f:
            results['Simple CNN'] = json.load(f)
    except:
        results['Simple CNN'] = {'test_accuracy': 0.8704}  # Your current result
    
    # MobileNet results
    try:
        with open('ml/outputs/mobilenet_results.json', 'r') as f:
            results['MobileNetV2'] = json.load(f)
    except:
        results['MobileNetV2'] = None
    
    # EfficientNet results
    try:
        with open('ml/outputs/efficientnet_results.json', 'r') as f:
            results['EfficientNetB0'] = json.load(f)
    except:
        results['EfficientNetB0'] = None
    
    return results

def plot_comparison(results):
    """Create comparison charts"""
    
    # Filter out None results
    valid_models = {k: v for k, v in results.items() if v is not None}
    
    if len(valid_models) == 0:
        print("No results to compare. Train models first!")
        return
    
    model_names = list(valid_models.keys())
    test_accuracies = [valid_models[m]['test_accuracy'] * 100 for m in model_names]
    val_accuracies = [valid_models[m].get('best_val_accuracy', 0) * 100 for m in model_names]
    
    # Create bar chart
    x = np.arange(len(model_names))
    width = 0.35
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Accuracy comparison
    bars1 = ax1.bar(x - width/2, test_accuracies, width, label='Test Accuracy', color='#2ecc71')
    bars2 = ax1.bar(x + width/2, val_accuracies, width, label='Validation Accuracy', color='#3498db')
    
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, fontsize=10)
    ax1.legend()
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    
    # Parameter comparison
    param_counts = []
    for model in model_names:
        if model == 'Simple CNN':
            param_counts.append(424268)  # From training output
        elif model == 'MobileNetV2':
            param_counts.append(2259844)  # Approximate
        elif model == 'EfficientNetB0':
            param_counts.append(4842927)  # Approximate
    
    colors = ['#e74c3c' if p == min(param_counts) else '#95a5a6' for p in param_counts]
    bars = ax2.bar(model_names, param_counts, color=colors)
    ax2.set_ylabel('Parameters (millions)', fontsize=12)
    ax2.set_title('Model Size Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Convert to millions
    for bar, count in zip(bars, param_counts):
        height = bar.get_height()
        ax2.annotate(f'{count/1e6:.1f}M', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('ml/outputs/model_comparison.png', dpi=150)
    plt.show()
    
    # Print summary table
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Model':<20} {'Test Acc':<12} {'Val Acc':<12} {'Parameters':<15} {'Recommendation':<15}")
    print("-" * 75)
    
    best_accuracy = max(test_accuracies)
    smallest_model = min(param_counts)
    
    for i, model in enumerate(model_names):
        test_acc = test_accuracies[i]
        val_acc = val_accuracies[i]
        params = param_counts[i]
        
        recommendation = ""
        if test_acc == best_accuracy:
            recommendation = "⭐ Best Accuracy"
        elif params == smallest_model:
            recommendation = "📱 Most Lightweight"
        else:
            recommendation = "Good Balance"
        
        print(f"{model:<20} {test_acc:.2f}%{' ':<9} {val_acc:.2f}%{' ':<9} {params/1e6:.1f}M{' ':<12} {recommendation}")
    
    print("\n" + "=" * 60)
    print("RECOMMENDATION:")
    print("=" * 60)
    
    # Find best model by accuracy
    best_idx = test_accuracies.index(max(test_accuracies))
    print(f"\n🏆 Best Accuracy: {model_names[best_idx]} with {test_accuracies[best_idx]:.1f}%")
    
    # Find smallest model
    smallest_idx = param_counts.index(min(param_counts))
    print(f"📱 Smallest Model: {model_names[smallest_idx]} with {param_counts[smallest_idx]/1e6:.1f}M parameters")
    
    print("\nFor web application deployment, consider:")
    print("  - Simple CNN: Fastest inference, good for CPU deployment")
    print("  - MobileNetV2: Good balance of accuracy and speed")
    print("  - EfficientNetB0: Best accuracy, requires more resources")
    
    return valid_models

def main():
    """Main comparison function"""
    print("=" * 60)
    print("📊 Model Comparison Tool")
    print("=" * 60)
    
    results = load_results()
    plot_comparison(results)
    
    print("\n✅ Comparison complete! Charts saved to ml/outputs/")

if __name__ == "__main__":
    main()