"""
Network Visualization Module

Creates interactive Plotly visualizations of similarity networks.
Edge width and darkness are proportional to similarity scores.
"""
import os
import json
from typing import Dict, Optional
from pathlib import Path
import argparse
import yaml

import numpy as np
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sys
sys.path.append(str(Path(__file__).parents[2]))
from utils.logging_setup import setup_logger
from utils.file_io import ensure_directory


class NetworkVisualizer:
    """
    Create interactive network visualizations
    """
    
    def __init__(
        self,
        config: dict,
        logger=None
    ):
        """
        Initialize visualizer
        
        Args:
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or setup_logger(__name__)
        
        # Extract visualization settings
        viz_config = config['visualization']
        self.node_size_range = tuple(viz_config['node_size_range'])
        self.edge_width_range = tuple(viz_config['edge_width_range'])
        self.color_map = viz_config['color_map']
        
        # Layout settings
        layout_config = config['layout']
        self.layout_algorithm = layout_config['algorithm']
        self.layout_seed = layout_config['seed']
        self.layout_iterations = layout_config['iterations']
        
        self.logger.info(f"Initialized network visualizer (layout={self.layout_algorithm}, seed={self.layout_seed})")
    
    def load_network(
        self,
        network_file: str
    ) -> nx.Graph:
        """
        Load network from JSON file
        
        Args:
            network_file: Path to network JSON file
        
        Returns:
            NetworkX graph
        """
        self.logger.info(f"Loading network from {network_file}")
        
        with open(network_file, 'r') as f:
            data = json.load(f)
        
        # Convert from node-link format
        G = nx.node_link_graph(data)
        
        self.logger.info(f"Loaded network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        return G
    
    def compute_layout(
        self,
        G: nx.Graph,
        algorithm: str = None
    ) -> Dict:
        """
        Compute network layout
        
        Args:
            G: NetworkX graph
            algorithm: Layout algorithm (spring, circular, etc.)
        
        Returns:
            Dictionary mapping node ID to (x, y) position
        """
        algorithm = algorithm or self.layout_algorithm
        
        self.logger.info(f"Computing {algorithm} layout...")
        
        if algorithm == 'spring':
            pos = nx.spring_layout(
                G,
                seed=self.layout_seed,
                iterations=self.layout_iterations
            )
        elif algorithm == 'circular':
            pos = nx.circular_layout(G)
        elif algorithm == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        else:
            raise ValueError(f"Unknown layout algorithm: {algorithm}")
        
        return pos
    
    def create_network_plot(
        self,
        G: nx.Graph,
        title: str = "Similarity Network",
        center_word: Optional[str] = None,
        output_file: Optional[str] = None
    ) -> go.Figure:
        """
        Create interactive Plotly network plot
        
        Args:
            G: NetworkX graph
            title: Plot title
            center_word: Optional word to highlight at center
            output_file: Optional file to save HTML
        
        Returns:
            Plotly figure
        """
        self.logger.info(f"Creating network plot: {title}")
        
        # Compute layout
        pos = self.compute_layout(G)
        
        # Prepare edge data
        edge_x = []
        edge_y = []
        edge_weights = []
        
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            weight = edge[2].get('weight', 1.0)
            edge_weights.append(weight)
        
        # Normalize edge weights for visualization
        if edge_weights:
            min_w, max_w = min(edge_weights), max(edge_weights)
            if max_w > min_w:
                edge_widths = [
                    self.edge_width_range[0] + (w - min_w) / (max_w - min_w) * (self.edge_width_range[1] - self.edge_width_range[0])
                    for w in edge_weights
                ]
            else:
                edge_widths = [self.edge_width_range[0]] * len(edge_weights)
        else:
            edge_widths = []
        
        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode='lines',
            line=dict(
                width=1,
                color='rgba(125, 125, 125, 0.3)'
            ),
            hoverinfo='none',
            showlegend=False
        )
        
        # Prepare node data
        node_x = []
        node_y = []
        node_text = []
        node_sizes = []
        node_colors = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Get label
            label = G.nodes[node].get('label', str(node))
            node_text.append(label)
            
            # Node size based on degree
            degree = G.degree(node)
            size = self.node_size_range[0] + (degree / max(dict(G.degree()).values())) * (self.node_size_range[1] - self.node_size_range[0])
            node_sizes.append(size)
            
            # Highlight center word if specified
            if center_word and label == center_word:
                node_colors.append('red')
            else:
                node_colors.append('skyblue')
        
        # Create node trace
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=node_text,
            textposition='top center',
            textfont=dict(size=8),
            hoverinfo='text',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=1, color='white')
            ),
            showlegend=False
        )
        
        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=title,
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='white',
                width=1200,
                height=800
            )
        )
        
        # Save if output file specified
        if output_file:
            ensure_directory(os.path.dirname(output_file))
            fig.write_html(output_file)
            self.logger.info(f"Saved plot to {output_file}")
        
        return fig
    
    def create_year_comparison(
        self,
        network_files: Dict[int, str],
        output_file: str,
        center_word: Optional[str] = None
    ):
        """
        Create subplot comparing networks across years
        
        Args:
            network_files: Dictionary mapping year to network file
            output_file: Output HTML file
            center_word: Optional word to highlight
        """
        years = sorted(network_files.keys())
        n_years = len(years)
        
        self.logger.info(f"Creating year comparison for {n_years} years")
        
        # Create subplots
        rows = (n_years + 2) // 3  # 3 columns
        cols = min(3, n_years)
        
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[f"Year {year}" for year in years],
            horizontal_spacing=0.05,
            vertical_spacing=0.1
        )
        
        for idx, year in enumerate(years):
            row = idx // 3 + 1
            col = idx % 3 + 1
            
            # Load and plot network
            G = self.load_network(network_files[year])
            pos = self.compute_layout(G)
            
            # Add traces (simplified for subplot)
            # ... (edge and node traces)
            
        ensure_directory(os.path.dirname(output_file))
        fig.write_html(output_file)
        self.logger.info(f"Saved comparison plot to {output_file}")


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description="Visualize similarity networks"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to network configuration YAML'
    )
    parser.add_argument(
        '--network',
        type=str,
        required=True,
        help='Path to network JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output HTML file'
    )
    parser.add_argument(
        '--title',
        type=str,
        default="Similarity Network",
        help='Plot title'
    )
    parser.add_argument(
        '--center-word',
        type=str,
        help='Word to highlight at center'
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create logger
    logger = setup_logger('network_visualization')
    
    # Create visualizer
    visualizer = NetworkVisualizer(config, logger)
    
    # Load network
    G = visualizer.load_network(args.network)
    
    # Create plot
    fig = visualizer.create_network_plot(
        G,
        title=args.title,
        center_word=args.center_word,
        output_file=args.output
    )
    
    logger.info("Visualization complete!")


if __name__ == '__main__':
    main()

