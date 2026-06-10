import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_box(ax, x, y, width, height, text, bg_color='#ffffff', text_color='black', fontsize=10):
    box = patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.2",
                                 edgecolor='#2c3e50', facecolor=bg_color, lw=2)
    ax.add_patch(box)
    
    # support multiline text
    ax.text(x + width/2, y + height/2, text, ha='center', va='center',
            color=text_color, fontsize=fontsize, weight='bold',
            wrap=True)
    return box

def draw_arrow(ax, x_start, y_start, x_end, y_end):
    ax.annotate("", xy=(x_end, y_end), xytext=(x_start, y_start),
                arrowprops=dict(arrowstyle="->", lw=2, color="#7f8c8d"))

def generate_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Inputs
    draw_box(ax, 0.5, 5.5, 2, 1, "R0 Data\n(Base)", bg_color='#d6eaf8')
    draw_box(ax, 0.5, 2.5, 2, 1, "R1 Data\n(Repairs)", bg_color='#d6eaf8')
    
    # Preprocessing
    draw_box(ax, 4, 4, 3, 1.5, "Preprocessing\n- Pivot & Merge\n- Target Scaling\n- One-Hot Encoding", bg_color='#fcf3cf')
    
    # Arrows to preproc
    draw_arrow(ax, 2.5, 6, 4, 4.75)
    draw_arrow(ax, 2.5, 3, 4, 4.75)
    
    # Models
    draw_box(ax, 8.5, 5.5, 2.5, 1.5, "Classical ML\n- XGBoost\n- Random Forest\n- LightGBM", bg_color='#d5f5e3')
    draw_box(ax, 8.5, 2.5, 2.5, 1.5, "Deep Learning\n- Transformer\n- CNN+LSTM\n- PINN", bg_color='#d5f5e3')
    
    # Arrows to models
    draw_arrow(ax, 7, 4.75, 8.5, 6.25)
    draw_arrow(ax, 7, 4.75, 8.5, 3.25)
    
    # Ensemble
    draw_box(ax, 12.5, 4, 2.5, 1.5, "Meta-Learner\n(Stack Ensembling)\nRidge Regressor", bg_color='#f5b7b1')
    
    # Arrows to ensemble
    draw_arrow(ax, 11, 6.25, 12.5, 4.75)
    draw_arrow(ax, 11, 3.25, 12.5, 4.75)
    
    plt.title("End-to-End Prediction Pipeline Architecture", fontsize=16, weight='bold', pad=10)
    plt.tight_layout()
    plt.savefig('pipeline_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_pinn_diagram():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis('off')
    
    draw_box(ax, 0.5, 2, 2, 1, "Scaled Features\n(Inputs)", bg_color='#ebdef0')
    draw_arrow(ax, 2.5, 2.5, 3.5, 2.5)
    
    draw_box(ax, 3.5, 1.5, 2.5, 2, "Neural Network\n(Dense Layers)\nReLU Activation", bg_color='#fadbd8')
    draw_arrow(ax, 6.0, 2.5, 7.0, 2.5)
    
    draw_box(ax, 7.0, 1.5, 2.5, 2, "Physics Constraint Loss\nCost Func Penalty:\nYS ≤ UTS", bg_color='#fdebd0')
    draw_arrow(ax, 9.5, 2.5, 10.5, 2.5)
    
    draw_box(ax, 10.5, 2, 1.5, 1, "R2 Outputs\n(Predictions)", bg_color='#d1f2eb')
    
    plt.title("Physics-Informed Neural Network (PINN) Logic Flow", fontsize=16, weight='bold', pad=10)
    plt.tight_layout()
    plt.savefig('pinn_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    generate_pipeline_diagram()
    generate_pinn_diagram()
