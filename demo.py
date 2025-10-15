import streamlit as st
import pandas as pd
import gravis as gv
import streamlit.components.v1 as components

from src.shared_components import get_current_team_data
from src.logic_visuals import churn_figs

# Configure page - must be first Streamlit command
st.set_page_config(
    page_title="Fanful MyTeam Analytics Demo",
    layout="wide"
)

# Password protection
def check_password():
    """Returns True if the user has entered the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["demo"]["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    # First run, show input for password
    if "password_correct" not in st.session_state:
        st.text_input(
            "Password",
            type="password",
            on_change=password_entered,
            key="password"
        )
        st.write("*Please enter the demo password to continue.*")
        return False
    # Password not correct, show input + error
    elif not st.session_state["password_correct"]:
        st.text_input(
            "Password",
            type="password",
            on_change=password_entered,
            key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct
        return True

# Check password before showing content
if check_password():
    # Main content
    st.title("Fanful MyTeam Analytics")
    
    # Get current team data
    team_data = get_current_team_data()
    analytics = team_data["analytics"]
    dfs = team_data["dfs"]
    
    # Network Graph Section
    with st.expander("🔗 Network Graph", expanded=True):
        st.caption(
            "Users are colored by network community. Grey users are considered latent. "
            "Scroll down for Top Communities and Users."
        )
        
        # Create and process network graph
        g = analytics.create_network_graph()
        g, coms = analytics.assign_network_properties(g)
        
        # Format node details for display on select
        for node, data in g.nodes(data=True):
            data["click"] = (
                f"Recency: {data['user_status']:.0f}\n"
                f"PageRank: {data['page_rank']:.2f}\n"
                f"Centrality: {data['centrality']:.2f}\n"
            )
        
        # Create network visualization
        network_figure = gv.d3(
            g,
            # visual settings
            graph_height=600,
            details_height=100,
            show_details=False,
            show_details_toggle_button=True,
            zoom_factor=0.4,
            # graph settings
            node_hover_neighborhood=True,
            node_hover_tooltip=False,
            use_node_size_normalization=True,
            node_size_normalization_max=100,
            node_size_data_source="size",
            use_edge_size_normalization=False,
            edge_size_data_source="size",
            edge_curvature=0.3,
            # layout settings
            large_graph_threshold=500,
            layout_algorithm_active=True,
            many_body_force_strength=-715,
            use_many_body_force_min_distance=True,
            many_body_force_min_distance=100.0,
            use_many_body_force_max_distance=True,
            many_body_force_max_distance=1000.0,
        )
        
        components.html(network_figure.to_html(), height=600, width=1000)
        
        st.divider()
        
        # Top Communities
        top_5_communities = sorted(coms, key=len, reverse=True)[:5]
        
        community_data = []
        for i, community in enumerate(top_5_communities):
            anchor = max(community, key=lambda node: g.degree[node])
            retention = round(
                len([user for user in community if g.nodes[user]["user_status"] < 30])
                / len(community)
                * 100,
                2,
            )
            
            community_data.append(
                {
                    "Community": i + 1,
                    "Anchor": anchor,
                    "Size": len(community),
                    "30-Day Retention": f"{retention}%",
                }
            )
        
        community_df = pd.DataFrame(community_data)
        st.write("**Top Communities**")
        st.dataframe(community_df, use_container_width=False, hide_index=True)
    
    # 30-Day Retention Curve Section
    with st.expander("📈 30-Day Retention Curve", expanded=False):
        st.caption("Retention analysis showing how users return over a 30-day period.")

        # Process retention data
        month_cohorts, month_rates = analytics.process_retention(
            dfs["engagement"], time_type="daily", backstop_days=240
        )
        
        # Create retention chart
        monthly_churn_fig = churn_figs(month_cohorts, month_rates, "daily", 240)
        st.plotly_chart(monthly_churn_fig, use_container_width=True)

    # Monthly Retention Curve Section
    with st.expander("📈 Monthly Retention Curve", expanded=False):
        st.caption("Retention analysis showing how users return over a monthly period.")
        
        # Process retention data
        month_cohorts, month_rates = analytics.process_retention(
            dfs["engagement"], time_type="monthly", backstop_days=240
        )
        
        # Create retention chart
        monthly_churn_fig = churn_figs(month_cohorts, month_rates, "monthly", 240)
        st.plotly_chart(monthly_churn_fig, use_container_width=True)
    