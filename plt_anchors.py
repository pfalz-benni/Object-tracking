import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Anchors (width, height)
anchors = [
            (83.2,  52.7),
            (245.0, 96.0),
            (56.9, 74.8),
            (54.4,  26.0),
            (77.74,  70.0)

]

# Plotting function
def plot_anchors(anchors):
    fig, ax = plt.subplots(figsize=(10, 10))
    colors = ['r', 'g', 'b', 'y', 'c']
    
    # Plot each anchor
    for i, (w, h) in enumerate(anchors):
        rect = patches.Rectangle((0, 0), w, h, linewidth=2, edgecolor=colors[i], facecolor='none', label=f'Anchor {i+1}')
        ax.add_patch(rect)
        
        # Annotate the width and height
        ax.text(w / 2, h / 2, f'{w}x{h}', fontsize=12, ha='center', va='center', color=colors[i])

    # Set limits based on the largest anchor
    max_width = max(anchor[0] for anchor in anchors)
    max_height = max(anchor[1] for anchor in anchors)
    
    ax.set_xlim(-1, max_width + 2)
    ax.set_ylim(-1, max_height + 2)
    
    # Add grid, legend, and labels
    ax.grid(True)
    ax.legend()
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    ax.set_title('Anchor Box Sizes')
    
    plt.show()

# Call the plotting function
plot_anchors(anchors)
