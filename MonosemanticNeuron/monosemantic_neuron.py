# %% Monosemantic Neuron Analysis with Sparse Autoencoder
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, sparsity_weight: float = 0.1):
        """Simple Sparse Autoencoder for neuron analysis
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layer (typically smaller than input_dim)
            sparsity_weight: Weight for sparsity penalty
        """
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.activation = nn.ReLU()
        self.sparsity_weight = sparsity_weight
        
    def forward(self, x):
        # Encode
        encoded = self.activation(self.encoder(x))
        # Add sparsity penalty
        sparsity_penalty = self.sparsity_weight * torch.mean(encoded)
        # Decode
        decoded = self.decoder(encoded)
        return decoded, sparsity_penalty
    
    def get_activations(self, x):
        """Get encoded activations for analysis"""
        with torch.no_grad():
            return self.activation(self.encoder(x))

class MonosemanticNeuronAnalyzer:
    def __init__(self, model_name: str = "gpt2-medium", layer_nums: List[int] = None):
        """Initialize analyzer with model and layers"""
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.layer_nums = layer_nums if layer_nums else list(range(4))
        self.activation_records = defaultdict(list)
        self.concept_map = defaultdict(dict)
        
        # Initialize SAE for each layer
        self.autoencoders = {}
        for layer_num in self.layer_nums:
            input_dim = self.model.config.hidden_size
            hidden_dim = input_dim // 2  # Reduce dimension by half
            self.autoencoders[layer_num] = SparseAutoencoder(input_dim, hidden_dim)
        
    def get_neuron_activations(self, text: str) -> Dict[int, torch.Tensor]:
        """Get neuron activations for input text"""
        tokens = self.tokenizer(text, return_tensors="pt")
        activations = {}
        
        def hook_fn(module, input, output, layer_num):
            activations[layer_num] = output[0].detach()
            
        hooks = []
        for layer_num in self.layer_nums:
            layer = self.model.transformer.h[layer_num]
            hook = layer.register_forward_hook(
                lambda mod, inp, out, ln=layer_num: hook_fn(mod, inp, out, ln)
            )
            hooks.append(hook)
            
        with torch.no_grad():
            self.model(**tokens)
            
        for hook in hooks:
            hook.remove()
            
        return activations
    
    def train_autoencoders(self, num_epochs: int = 10, batch_size: int = 32):
        """Train sparse autoencoders on collected activations"""
        print("Training autoencoders for each layer...")
        for layer_num in self.layer_nums:
            print(f"Training layer {layer_num}...")
            # Collect all activations for this layer
            layer_acts = []
            for record in self.activation_records[layer_num]:
                layer_acts.append(record['activations'])
            layer_acts = torch.cat(layer_acts, dim=0)
            
            # Train autoencoder
            optimizer = torch.optim.Adam(self.autoencoders[layer_num].parameters())
            for epoch in range(num_epochs):
                total_loss = 0
                # Process in batches
                for i in range(0, len(layer_acts), batch_size):
                    batch = layer_acts[i:i+batch_size]
                    optimizer.zero_grad()
                    reconstructed, sparsity_penalty = self.autoencoders[layer_num](batch)
                    loss = nn.MSELoss()(reconstructed, batch) + sparsity_penalty
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                if (epoch + 1) % 2 == 0:
                    print(f"Epoch {epoch+1}, Loss: {total_loss/len(layer_acts):.4f}")
    
    def analyze_concept(self, concept_name: str, examples: List[str]):
        """Analyze neuron responses to concept examples"""
        for example in examples:
            activations = self.get_neuron_activations(example)
            for layer_num, layer_activations in activations.items():
                mean_activations = layer_activations.mean(dim=1)
                self.activation_records[layer_num].append({
                    'concept': concept_name,
                    'text': example,
                    'activations': mean_activations
                })
    
    def compute_selectivity(self, threshold: float = 0.9) -> Dict[int, List[int]]:
        """Compute neuron selectivity scores using SAE features"""
        monosemantic_neurons = {}
        self.selectivity_scores = {}
        
        # First train the autoencoders
        self.train_autoencoders()
        
        for layer_num in self.layer_nums:
            layer_records = self.activation_records[layer_num]
            concept_activations = defaultdict(list)
            
            # Get SAE encoded features
            for record in layer_records:
                acts = record['activations']
                encoded_acts = self.autoencoders[layer_num].get_activations(acts)
                concept_activations[record['concept']].append(encoded_acts.numpy())
            
            n_neurons = concept_activations[list(concept_activations.keys())[0]][0].shape[-1]
            selectivity_scores = np.zeros(n_neurons)
            
            for neuron_idx in range(n_neurons):
                max_mean_activation = 0
                total_mean_activation = 0
                
                for concept, activations in concept_activations.items():
                    mean_act = np.mean([act[0, neuron_idx] for act in activations])
                    max_mean_activation = max(max_mean_activation, mean_act)
                    total_mean_activation += mean_act
                
                if total_mean_activation > 0:
                    selectivity_scores[neuron_idx] = max_mean_activation / total_mean_activation
            
            self.selectivity_scores[layer_num] = selectivity_scores
            monosemantic_neurons[layer_num] = np.where(selectivity_scores > threshold)[0].tolist()
            
        return monosemantic_neurons

    def visualize_selectivity(self, save_path: Optional[str] = None):
        """Visualize neuron selectivity scores"""
        if not hasattr(self, 'selectivity_scores'):
            raise ValueError("Must run compute_selectivity() before visualization")
            
        n_layers = len(self.layer_nums)
        fig, axes = plt.subplots(n_layers, 1, figsize=(15, 4*n_layers))
        if n_layers == 1:
            axes = [axes]
            
        for ax, layer_num in zip(axes, self.layer_nums):
            scores = self.selectivity_scores[layer_num]
            sns.histplot(scores, bins=50, ax=ax)
            ax.set_title(f'Layer {layer_num} SAE-Encoded Neuron Selectivity Distribution')
            ax.set_xlabel('Selectivity Score')
            ax.set_ylabel('Count')
            
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def visualize_mean_activations(self, layer_num: int, save_path: Optional[str] = None):
        """Visualize mean activations for concepts using SAE features"""
        layer_records = self.activation_records[layer_num]
        concept_means = defaultdict(list)
        
        # Get SAE encoded features
        for record in layer_records:
            acts = record['activations']
            encoded_acts = self.autoencoders[layer_num].get_activations(acts)
            concept_means[record['concept']].append(encoded_acts.numpy())
        
        mean_activations = {concept: np.mean(np.array(acts), axis=0) for concept, acts in concept_means.items()}
        plt.figure(figsize=(12, 6))
        for concept, activations in mean_activations.items():
            activations = activations.flatten()
            plt.bar(range(len(activations)), activations, label=concept, alpha=0.5)
        
        plt.title(f'SAE-Encoded Mean Activations for Layer {layer_num}')
        plt.xlabel('Encoded Neuron Index')
        plt.ylabel('Mean Activation')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def visualize_concept_activations(self, layer_num: int, top_n_neurons: int = 10, save_path: Optional[str] = None):
        """Visualize word-level activation patterns using SAE features"""
        layer_records = self.activation_records[layer_num]
        word_neuron_activations = defaultdict(list)
        
        # Extract key words from each example
        for record in layer_records:
            # Get the main word from the example (e.g., "happy", "blue", "running")
            text = record['text'].lower()
            words = text.split()
            key_words = []
            
            # Extract relevant words based on the concept
            if record['concept'] == 'emotions':
                emotion_words = ['happy', 'angry', 'worried', 'sad', 'joy']
                key_words = [w for w in words if w in emotion_words]
            elif record['concept'] == 'colors':
                color_words = ['blue', 'red', 'green']
                key_words = [w for w in words if w in color_words]
            elif record['concept'] == 'numbers':
                # Extract numeric words and numbers
                number_words = ['seven', 'hundred', 'thirteen', 'fifty', 'twenty']
                key_words = [w for w in words if w in number_words or w.isdigit()]
            elif record['concept'] == 'actions':
                action_words = ['running', 'flying', 'walks']
                key_words = [w for w in words if w in action_words]
            
            if key_words:
                word = key_words[0]  # Take the first relevant word
                acts = record['activations']
                encoded_acts = self.autoencoders[layer_num].get_activations(acts)
                word_neuron_activations[word].append(encoded_acts.numpy()[0])
        
        # Calculate mean activations for each word
        mean_activations = {word: np.mean(activations, axis=0) 
                          for word, activations in word_neuron_activations.items()}
        
        # Create DataFrame for visualization
        df_data = pd.DataFrame(mean_activations).T
        
        # Find neurons with highest variance
        neuron_vars = df_data.var()
        top_neurons = neuron_vars.nlargest(top_n_neurons).index
        df_plot = df_data[top_neurons]
        
        # Plot heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(df_plot, annot=True, cmap='RdBu_r', center=0, fmt='.2f')
        plt.title(f'Layer {layer_num} Top {top_n_neurons} SAE-Encoded Neurons:\nWord-Level Activations')
        plt.xlabel('Encoded Neuron Index')
        plt.ylabel('Word')
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
        # Print detailed analysis
        print(f"\nLayer {layer_num} Word-Level Analysis:")
        print("="*50)
        for neuron_idx in top_neurons:
            print(f"\nNeuron {neuron_idx}:")
            # Sort words by activation strength for this neuron
            word_activations = [(word, acts[neuron_idx]) 
                              for word, acts in mean_activations.items()]
            word_activations.sort(key=lambda x: abs(x[1]), reverse=True)
            
            print("Top activating words:")
            for word, activation in word_activations[:3]:
                print(f"  {word}: {activation:.3f}")

    def visualize_monosemantic_neurons(self, layer_num: int, top_k: int = 5):
        """Visualize activation patterns of top monosemantic neurons across different words"""
        # Get encoded activations for each word
        layer_records = self.activation_records[layer_num]
        word_neuron_activations = defaultdict(list)
        
        # Process each example to get word-level activations
        for record in layer_records:
            text = record['text'].lower()
            # Extract the main word from the text
            main_word = None
            if "happy" in text or "angry" in text or "sad" in text or "joy" in text:
                main_word = next(word for word in ["happy", "angry", "sad", "joy"] if word in text)
            elif "blue" in text or "red" in text or "green" in text:
                main_word = next(word for word in ["blue", "red", "green"] if word in text)
            elif "running" in text or "flying" in text or "walks" in text:
                main_word = next(word for word in ["running", "flying", "walks"] if word in text)
            
            if main_word:
                acts = record['activations']
                encoded_acts = self.autoencoders[layer_num].get_activations(acts)
                word_neuron_activations[main_word].append(encoded_acts.numpy()[0])
        
        # Calculate mean activations for each word
        mean_activations = {word: np.mean(activations, axis=0) 
                          for word, activations in word_neuron_activations.items()}
        
        # Find most selective neurons
        n_neurons = next(iter(mean_activations.values())).shape[0]
        selectivity_scores = []
        
        for neuron_idx in range(n_neurons):
            neuron_acts = {word: acts[neuron_idx] for word, acts in mean_activations.items()}
            max_act = max(neuron_acts.values())
            total_act = sum(abs(act) for act in neuron_acts.values())
            selectivity = max_act / total_act if total_act > 0 else 0
            selectivity_scores.append((neuron_idx, selectivity))
        
        # Get top-k selective neurons
        top_neurons = sorted(selectivity_scores, key=lambda x: x[1], reverse=True)[:top_k]
        
        # Visualize each top neuron's activation pattern
        for neuron_idx, selectivity in top_neurons:
            plt.figure(figsize=(15, 5))
            
            # Get activations for this neuron across all words
            word_acts = [(word, acts[neuron_idx]) for word, acts in mean_activations.items()]
            word_acts.sort(key=lambda x: x[1], reverse=True)
            
            words, activations = zip(*word_acts)
            
            # Create bar plot
            bars = plt.bar(words, activations)
            
            # Color coding
            for bar, activation in zip(bars, activations):
                if activation > 0:
                    bar.set_color('red')
                else:
                    bar.set_color('blue')
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom' if height > 0 else 'top')
            
            plt.title(f'Layer {layer_num}, Neuron {neuron_idx} Activations\nSelectivity Score: {selectivity:.3f}')
            plt.xlabel('Words')
            plt.ylabel('Activation Value')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            
            # Add horizontal line at y=0
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            # Print analysis
            print(f"\nNeuron {neuron_idx} Analysis:")
            print(f"Selectivity Score: {selectivity:.3f}")
            print("Top activating words:")
            for word, act in word_acts[:3]:
                print(f"  {word}: {act:.3f}")
            print("Lowest activating words:")
            for word, act in word_acts[-3:]:
                print(f"  {word}: {act:.3f}")

    def visualize_neuron_activations(self, layer_num: int, neuron_idx: int):
        """Visualize how a specific neuron responds to different words"""
        # Test words from different categories
        test_words = [
            "cat", "dog", "lion", "tiger",  # animals
            "apple", "banana", "orange",     # fruits
            "car", "train",                  # vehicles
            "happy", "sad", "angry",         # emotions
            "red", "blue", "green",          # colors
            "run", "walk", "jump"            # actions
        ]
        
        # Get activations for each word
        word_activations = []
        
        for word in test_words:
            # Get model activations
            tokens = self.tokenizer(word, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(tokens.input_ids, output_hidden_states=True)
            
            # Get the activation for the target layer
            layer_output = outputs.hidden_states[layer_num]
            
            # Get activation for the word token (index 1, after [CLS])
            activation = layer_output[0, 1]
            
            # Get encoded activation using SAE
            encoded_activation = self.autoencoders[layer_num].get_activations(activation.unsqueeze(0))
            
            # Store the activation value for our target neuron
            word_activations.append(encoded_activation[0, neuron_idx].item())
        
        # Create visualization
        plt.figure(figsize=(15, 6))
        bars = plt.bar(test_words, word_activations)
        
        # Color code the bars
        for bar, activation in zip(bars, word_activations):
            if activation > 0:
                bar.set_color('red')
            else:
                bar.set_color('blue')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=8)
        
        plt.title(f'Layer {layer_num}, Neuron {neuron_idx} Activation Values Across Words')
        plt.xlabel('Words')
        plt.ylabel('Activation Value')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Add horizontal line at y=0
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print analysis
        print(f"\nNeuron {neuron_idx} Analysis:")
        word_acts = list(zip(test_words, word_activations))
        word_acts.sort(key=lambda x: abs(x[1]), reverse=True)
        
        print("\nStrongest activations (absolute value):")
        for word, act in word_acts[:5]:
            print(f"  {word}: {act:.3f}")
        
        # Calculate selectivity
        max_act = max(abs(act) for act in word_activations)
        total_act = sum(abs(act) for act in word_activations)
        selectivity = max_act / total_act if total_act > 0 else 0
        
        print(f"\nSelectivity score: {selectivity:.3f}")
        
        # Try to identify the concept this neuron might be detecting
        positive_acts = [(w, a) for w, a in word_acts if a > 0]
        negative_acts = [(w, a) for w, a in word_acts if a < 0]
        
        print("\nPossible concept detection:")
        if positive_acts:
            print("Positively activated by:", ", ".join(w for w, _ in positive_acts[:3]))
        if negative_acts:
            print("Negatively activated by:", ", ".join(w for w, _ in negative_acts[:3]))

# Run analysis when script is executed
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = MonosemanticNeuronAnalyzer(
        model_name="gpt2-medium",
        layer_nums=[0, 1, 2, 3]
    )
    
    # Define diverse concepts and examples
    concepts = {
        "numbers": [
            "There are seven apples in this box.",
            "He ran 100 meters in 13 seconds.",
            "The year 2024 will bring new beginnings."
        ],
        "emotions": [
            "I was so happy that tears came to my eyes.",
            "His words made me extremely angry.",
            "I'm really worried about the test results."
        ],
        "actions": [
            "He is running quickly across the field.",
            "Birds are flying through the sky.",
            "The cat walks gracefully across the room."
        ],
        "colors": [
            "The sky is a beautiful shade of blue.",
            "The roses in the garden are bright red.",
            "Fresh grass has a vibrant green color."
        ]
    }
    
    # Analyze concepts
    for concept, examples in concepts.items():
        analyzer.analyze_concept(concept, examples)
    
    # Compute selectivity and prepare visualizations
    monosemantic_neurons = analyzer.compute_selectivity(threshold=0.6)
    
    # Set plot style
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    for i, layer_num in enumerate(analyzer.layer_nums):
        plt.subplot(2, 2, i+1)
        scores = analyzer.selectivity_scores[layer_num]
        sns.histplot(scores, bins=30, kde=True)
        plt.title(f'Layer {layer_num} SAE-Encoded Neuron Selectivity\nMonosemantic Threshold = 0.6')
        plt.xlabel('Selectivity Score')
        plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('portfolio_selectivity_dist.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Visualize concept activations and mean activations for each layer
    for layer_num in analyzer.layer_nums:
        analyzer.visualize_concept_activations(
            layer_num=layer_num,
            top_n_neurons=8,
            save_path=f'portfolio_concept_activations_layer_{layer_num}.png'
        )
        analyzer.visualize_mean_activations(
            layer_num=layer_num,
            save_path=f'portfolio_mean_activations_layer_{layer_num}.png'
        )
    
    # Print analysis summary
    print("\nMonosemantic Neuron Analysis Summary (using Sparse Autoencoder):")
    print("==========================================================")
    for layer_num, neurons in monosemantic_neurons.items():
        print(f"\nLayer {layer_num}:")
        print(f"- Found {len(neurons)} monosemantic neurons in encoded space")
        print(f"- Encoded neuron indices: {neurons[:10]}...")
        avg_selectivity = np.mean(analyzer.selectivity_scores[layer_num])
        print(f"- Average selectivity score: {avg_selectivity:.3f}")

