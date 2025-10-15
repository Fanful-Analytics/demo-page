import streamlit as st
import pandas as pd
import plotly.express as px
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
    
    # Key Metrics Section
    with st.expander("📊 Key Metrics", expanded=True):
        # Calculate metrics
        total_users = len(dfs["user"])
        engagement_df = dfs["engagement"].copy()
        sessions_df = dfs["sessions"].copy()
        
        # Prepare engagement data with proper datetime
        engagement_df["datestamp"] = pd.to_datetime(engagement_df["datestamp"])
        user_activity = engagement_df[["sender_id", "datestamp"]].drop_duplicates()
        user_activity = user_activity.set_index("datestamp").sort_index()
        
        # Function to count unique users in a rolling window (from dau_line_chart)
        def count_unique_users_in_window(activity_df, day_window):
            counts = {}
            for day in activity_df.index.unique():
                window_start = day - pd.Timedelta(days=day_window - 1)
                mask = (activity_df.index >= window_start) & (activity_df.index <= day)
                window_data = activity_df[mask]
                counts[day] = window_data["sender_id"].nunique()
            return pd.Series(counts)
        
        # Calculate rolling DAU
        rolling_1d = count_unique_users_in_window(user_activity, 1)
        rolling_7d = count_unique_users_in_window(user_activity, 7)
        rolling_30d = count_unique_users_in_window(user_activity, 30)
        
        # Get latest values and calculate percentages
        dau_1d = rolling_1d.iloc[-1] if len(rolling_1d) > 0 else 0
        dau_7d = rolling_7d.iloc[-1] if len(rolling_7d) > 0 else 0
        dau_30d = rolling_30d.iloc[-1] if len(rolling_30d) > 0 else 0
        
        monthly_pct = (dau_30d / total_users * 100) if total_users > 0 else 0
        weekly_pct = (dau_7d / total_users * 100) if total_users > 0 else 0
        daily_pct = (dau_1d / total_users * 100) if total_users > 0 else 0
        
        # Session metrics with 7-day rolling average per user
        sessions_df["session_date"] = pd.to_datetime(sessions_df["session_start"]).dt.date
        sessions_df["session_date"] = pd.to_datetime(sessions_df["session_date"])
        
        # Rolling 7-day average sessions per user
        def calc_rolling_sessions(sessions_df, day_window=7):
            daily_sessions = sessions_df.groupby("session_date").size()
            daily_users = sessions_df.groupby("session_date")["user_id"].nunique()
            sessions_per_user = daily_sessions / daily_users
            return sessions_per_user.rolling(window=day_window, min_periods=1).mean().iloc[-1] if len(sessions_per_user) > 0 else 0
        
        # Rolling 7-day average minutes per user
        def calc_rolling_minutes(sessions_df, day_window=7):
            controlled = sessions_df[sessions_df["session_length_minutes"] < 120]
            daily_minutes = controlled.groupby("session_date")["session_length_minutes"].sum()
            daily_users = controlled.groupby("session_date")["user_id"].nunique()
            minutes_per_user = daily_minutes / daily_users
            return minutes_per_user.rolling(window=day_window, min_periods=1).mean().iloc[-1] if len(minutes_per_user) > 0 else 0
        
        # Rolling 7-day average interactions per user
        def calc_rolling_interactions(engagement_df, day_window=7):
            daily_interactions = engagement_df.groupby("datestamp").size()
            daily_users = engagement_df.groupby("datestamp")["sender_id"].nunique()
            interactions_per_user = daily_interactions / daily_users
            return interactions_per_user.rolling(window=day_window, min_periods=1).mean().iloc[-1] if len(interactions_per_user) > 0 else 0
        
        avg_sessions_per_user = calc_rolling_sessions(sessions_df)
        avg_minutes_per_user = calc_rolling_minutes(sessions_df)
        avg_interactions_per_user = calc_rolling_interactions(engagement_df)
        
        # Row 1: User Activity Rates (All Downloads)
        st.markdown("### User Activity Rates (All Downloads)")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Monthly Active Users %",
                f"{monthly_pct:.1f}%",
                help="30-day rolling unique active users / Total users (all downloads)"
            )
        
        with col2:
            st.metric(
                "Weekly Active Users %",
                f"{weekly_pct:.1f}%",
                help="7-day rolling unique active users / Total users (all downloads)"
            )
        
        with col3:
            st.metric(
                "Daily Active Users %",
                f"{daily_pct:.1f}%",
                help="Daily unique active users / Total users (all downloads)"
            )
        
        st.divider()
        
        # Row 2: Engagement Metrics
        st.markdown("### Engagement Metrics (7-Day Rolling Avg per User)")
        col4, col5, col6 = st.columns(3)
        
        with col4:
            st.metric(
                "Avg Daily Sessions/User",
                f"{avg_sessions_per_user:.2f}",
                help="7-day rolling average: Daily sessions / Daily unique users"
            )
        
        with col5:
            st.metric(
                "Avg Daily Minutes/User",
                f"{avg_minutes_per_user:.1f}",
                help="7-day rolling average: Daily minutes / Daily unique users (sessions capped at 120 min)"
            )
        
        with col6:
            st.metric(
                "Avg Daily Interactions/User",
                f"{avg_interactions_per_user:.1f}",
                help="7-day rolling average: Daily interactions / Daily unique users"
            )
        
        st.divider()
        
        # Row 3: Subscription Metrics
        st.markdown("### Subscription Metrics")
        col7, col8, col9 = st.columns(3)
        
        # Calculate subscription metrics
        subscriptions_df = dfs["subscriptions"]
        unique_subscribers = subscriptions_df["user_id"].nunique()
        
        # Get active users count
        user_df = dfs["user"]
        active_users = len(user_df[user_df["user_status"] == "Active"])
        subscription_pct = (unique_subscribers / active_users * 100) if active_users > 0 else 0
        
        with col7:
            st.metric(
                "% Active Users Subscribed",
                f"{subscription_pct:.1f}%",
                help="Unique subscribers / Active users"
            )
        
        with col8:
            st.metric(
                "Mnthly Cost of Premium Subscription",
                "$4.99",
                help="Monthly cost for premium subscription"
            )
        
        with col9:
            st.metric(
                "% Subscribed to MVP (Premium) Tier",
                "100%",
                help="Percentage of subscribers on MVP (top tier)"
            )
    
    # RFMP Analysis Section
    with st.expander("📊 RFMP Analysis", expanded=False):
        st.caption("Recency, Frequency, Monetary, and Points analysis of user engagement")
        
        rfmp_data = dfs["rfmp"].copy()
        
        # RFMP Segmentation Charts
        st.write("**RFMP Segmentation by User Status**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**RFMP Segment (% Distribution)**")
            # Group by rfmp_segment and user_status
            segment_status = rfmp_data.groupby(["rfmp_segment", "user_status"], observed=True).size().reset_index(name="count")
            
            # Calculate percentages within each segment
            segment_totals = segment_status.groupby("rfmp_segment", observed=True)["count"].transform("sum")
            segment_status["percentage"] = (segment_status["count"] / segment_totals * 100).round(1)
            
            fig = px.bar(
                segment_status,
                x="rfmp_segment",
                y="percentage",
                color="user_status",
                labels={"rfmp_segment": "RFMP Segment", "percentage": "Percentage (%)", "user_status": "User Status"},
                category_orders={"rfmp_segment": ["Bronze", "Silver", "Gold"]},
                color_discrete_map={
                    "Active": "#2ecc71",
                    "Latent": "#f39c12", 
                    "Churned": "#e74c3c",
                    "Inactive": "#95a5a6"
                },
                custom_data=["percentage"]
            )
            fig.update_traces(
                hovertemplate="<b>%{fullData.name}</b><br>RFMP Segment: %{x}<br>Percentage: %{customdata[0]:.1f}%<extra></extra>"
            )
            fig.update_layout(
                height=400, 
                showlegend=True, 
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                yaxis=dict(range=[0, 100], ticksuffix="%"),
                barmode="stack"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**RFMP Score Distribution**")
            # Group by rfmp_score and user_status
            score_status = rfmp_data.groupby(["rfmp_score", "user_status"], observed=True).size().reset_index(name="count")
            
            # Calculate total for percentages
            total_users = score_status["count"].sum()
            score_status["percentage"] = (score_status["count"] / total_users * 100).round(1)
            
            fig = px.bar(
                score_status,
                x="rfmp_score",
                y="count",
                color="user_status",
                labels={"rfmp_score": "RFMP Score", "count": "User Count", "user_status": "User Status"},
                color_discrete_map={
                    "Active": "#2ecc71",
                    "Latent": "#f39c12",
                    "Churned": "#e74c3c",
                    "Inactive": "#95a5a6"
                },
                custom_data=["percentage"]
            )
            fig.update_traces(
                hovertemplate="<b>%{fullData.name}</b><br>RFMP Score: %{x}<br>Percentage: %{customdata[0]:.1f}%<extra></extra>"
            )
            fig.update_layout(
                height=400, 
                showlegend=True, 
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                yaxis=dict(showticklabels=False, title="")
            )
            st.plotly_chart(fig, use_container_width=True)
    
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
    