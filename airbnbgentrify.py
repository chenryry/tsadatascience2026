import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Set up the figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Map 1: Airbnb intensity
gdf_final.plot(
    column='listings_per_1000_residents',
    ax=ax1,
    legend=True,
    cmap='Reds',
    edgecolor='white',
    linewidth=0.2,
    missing_kwds={'color': 'lightgrey', 'label': 'No data'}
)
ax1.set_title('Airbnb Listings per 1,000 Residents', fontsize=14, fontweight='bold')
ax1.axis('off')

# Map 2: Gentrification proxy
gdf_final.plot(
    column='gentrification_proxy',
    ax=ax2,
    legend=True,
    cmap='Blues',
    edgecolor='white',
    linewidth=0.2,
    missing_kwds={'color': 'lightgrey', 'label': 'No data'}
)
ax2.set_title('Gentrification Proxy Index', fontsize=14, fontweight='bold')
ax2.axis('off')

plt.tight_layout()
plt.suptitle('Tourism Pressure vs Neighborhood Change in NYC', fontsize=16, y=1.02)
plt.savefig('hook_split_map.png', dpi=300, bbox_inches='tight')
plt.show()