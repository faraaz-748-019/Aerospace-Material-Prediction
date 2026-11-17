"""
Generate the high-resolution front cover image tailored specifically for the
M250 Maraging Steel Weld Repair Property Predictor project, highlighting Faisal Faraaz Syed.
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_cover_page_image():
    # 8.5 x 11 inches at 300 dpi
    fig, ax = plt.subplots(figsize=(8.5, 11), dpi=300)
    ax.set_xlim(0, 8.5)
    ax.set_ylim(0, 11)
    ax.axis('off')
    
    # Pure white background
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    
    # 1. Dark Blue Card
    # Dimensions: left=1.05, width=6.4, bottom=5.15, height=3.5
    card_x = 1.05
    card_y = 5.15
    card_w = 6.4
    card_h = 3.5
    
    card = patches.Rectangle((card_x, card_y), card_w, card_h,
                             facecolor='#1C355E', edgecolor='none')
    ax.add_patch(card)
    
    # Inside Card Text:
    # "PROJECT REPORT" - teal/cyan tracked
    ax.text(card_x + 0.45, card_y + card_h - 0.55, "P R O J E C T   R E P O R T",
            color='#5BC0BE', fontsize=10.5, weight='bold', fontfamily='sans-serif')
    
    # Main Title cleanly balanced
    title_text = "M250 Maraging Steel:\nPost-Weld Repair Property &\nStress-Strain Predictor"
    ax.text(card_x + 0.45, card_y + card_h - 1.75, title_text,
            color='#FFFFFF', fontsize=18.5, weight='bold', fontfamily='sans-serif',
            linespacing=1.22)
    
    # Subtitle cleanly formatted on two balanced lines
    sub_text = "Physics-Informed Neural Networks, Stacking Ensembles & XAI\nfor Aerospace Launch Vehicle Alloys (VSSC / ISRO)"
    ax.text(card_x + 0.45, card_y + 0.42, sub_text,
            color='#E0E6ED', fontsize=10.5, style='italic', fontfamily='sans-serif',
            linespacing=1.3)
    
    # 2. Strategic / Bottom-up Research Concept
    ax.text(1.05, 4.52, "Strategic Aerospace Research Initiative",
            color='#1C355E', fontsize=12.5, weight='bold', fontfamily='sans-serif')
    ax.text(1.05, 4.22, "Target Application: TIG-welded thin-shell booster motorcases (LSLF Coupons — S139 / S200)",
            color='#64748B', fontsize=10.5, fontfamily='sans-serif')
    
    # 3. Prepared By (Author Block - Highlighting User)
    ax.text(1.05, 3.45, "P R E P A R E D   B Y",
            color='#0D9488', fontsize=9, weight='bold', fontfamily='sans-serif')
    
    ax.text(1.05, 3.02, "Faisal Faraaz Syed",
            color='#0F172A', fontsize=18.5, weight='bold', fontfamily='sans-serif')
    
    ax.text(1.05, 2.70, "B.E. Computer Science & Engineering (AI & ML)",
            color='#334155', fontsize=10.5, fontfamily='sans-serif')
    ax.text(1.05, 2.42, "Vasavi College of Engineering, Hyderabad",
            color='#334155', fontsize=10.5, fontfamily='sans-serif')
    
    # 4. Divider Line
    ax.plot([1.05, 7.45], [1.70, 1.70], color='#CBD5E1', lw=1.0)
    
    # 5. Footer Metadata
    ax.text(1.05, 1.42, "Technical Project Report  •  Version 1.0  •  October 2026  •  Aerospace Materials Prediction",
            color='#64748B', fontsize=8.5, fontfamily='sans-serif')
    ax.text(1.05, 1.18, "Vikram Sarabhai Space Centre (VSSC / ISRO) Materials Qualification Program",
            color='#64748B', fontsize=8.5, style='italic', fontfamily='sans-serif')
    
    out_path = 'docs/assets/cover_page.png'
    os.makedirs('docs/assets', exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()
    print("Clean tailored cover image saved to:", out_path)

if __name__ == '__main__':
    generate_cover_page_image()
