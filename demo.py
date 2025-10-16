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

# Check if RFMP tab should be shown
SHOW_RFMP = st.secrets.get("RFMP", True)

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
                "32.1%",
                help="30-day rolling unique active users / Total users (all downloads)"
            )
        
        with col2:
            st.metric(
                "Weekly Active Users %",
                "22.2%",
                help="7-day rolling unique active users / Total users (all downloads)"
            )
        
        with col3:
            st.metric(
                "Daily Active Users %",
                "15.0%",
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
    
    # RFMP Analysis Section with Tabs
    with st.expander("📊 Engagement", expanded=False):
        # Create tabs based on RFMP secret
        if SHOW_RFMP:
            tab0, tab1 = st.tabs(["Topline Metrics", "RFMP"])
        else:
            tab0 = st.tabs(["Topline Metrics"])[0]
            tab1 = None
        
        with tab0:
            st.write("**Topline Metrics**")
            
            # Calculate metrics
            metrics = analytics.get_topline_metrics(
                dfs["user"], dfs["engagement"], dfs["subscriptions"]
            )
            
            # 3 columns for metrics
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                # Total sessions
                st.metric(
                    "Total Sessions",
                    f"{len(dfs['sessions']):,}",
                )
                
                # Sessions per user
                st.metric(
                    "Sessions per User",
                    f"{len(dfs['sessions']) / metrics['users']['total']:.2f}",
                )
                
                controlled_sessions = dfs["sessions"][
                    dfs["sessions"]["session_length_minutes"] < 120
                ]
                
                # Average session duration
                st.metric(
                    "Average Session Duration",
                    f"{controlled_sessions['session_length_minutes'].mean():.2f} minutes",
                )
                
                # Total session duration
                total_session_duration_mins = controlled_sessions[
                    "session_length_minutes"
                ].sum()
                total_session_duration_hours = total_session_duration_mins / 60
                total_session_duration_days = total_session_duration_hours / 24
                st.metric(
                    "Total Session Duration*",
                    f"{total_session_duration_mins:,} minutes",
                )
            
            with col2:
                # Total engagements
                st.metric(
                    "Total Engagements",
                    f"{len(dfs['engagement']):,}",
                )
                
                # Engagements per user
                st.metric(
                    "Engagements per User",
                    f"{len(dfs['engagement']) / metrics['users']['total']:.2f}",
                )
                
                # Engagement to reward ratio
                engagements_with_points = dfs["engagement"][dfs["engagement"]["points"] > 0]
                st.metric(
                    "Engagements per Reward",
                    f'{len(dfs["engagement"]) / len(engagements_with_points):.2f}',
                )
                
                # Total session duration in hours
                st.metric(
                    "Total Session Duration",
                    f"{total_session_duration_hours:.2f} hours",
                )
            
            with col3:
                # Total rewards
                st.metric(
                    "Total Rewards",
                    f"{len(engagements_with_points):,}",
                )
                
                # Rewards per user
                st.metric(
                    "Rewards per User",
                    f"{len(dfs['rewards']) / metrics['users']['total']:.2f}",
                )
                
                # Total reward points
                st.metric(
                    "Total Reward Points",
                    f"{engagements_with_points['points'].sum():,}",
                )
                
                # Total session duration in days
                st.metric(
                    "Total Session Duration",
                    f"{total_session_duration_days:.2f} days",
                )
            
            st.divider()
            st.markdown(
                "**Sessions that lasted longer than 2 hours were excluded from the total session duration and average session duration metrics.*"
            )
        
        if tab1 is not None:
            with tab1:
                rfmp_data = dfs["rfmp"].copy()
                
                with st.expander("Recency (how recently a customer has had a session)", expanded=True):
                    # Make a distribution plot of the recency, with color gradient green to red
                    recency_counts = rfmp_data.groupby("recency").size().reset_index(name="count")
                    recency_count_60 = sum(recency_counts[recency_counts["recency"] > 60]["count"])
                    recency_count_30 = sum(
                        recency_counts[
                            (recency_counts["recency"] >= 30) & (recency_counts["recency"] <= 60)
                        ]["count"]
                    )
                    recency_count_10 = sum(recency_counts[recency_counts["recency"] <= 10]["count"])
                    # Limit recency to less than or equal to 60
                    recency_counts = recency_counts[recency_counts["recency"] <= 60]
                    
                    # Three columns for the three metrics
                    col1, col2, col3 = st.columns([1, 1, 1])
                    
                    with col1:
                        st.success(f"**R <= 10: {recency_count_10:,}**")
                    
                    with col2:
                        st.warning(f"**Latent: {recency_count_30:,}**")
                    
                    with col3:
                        st.error(f"**Latent > 60: {recency_count_60:,}**")
                    
                    # Create a bar chart of the recency counts
                    fig = px.bar(
                        recency_counts,
                        x="recency",
                        y="count",
                        color="recency",
                        color_continuous_scale="rdylgn_r",
                    )
                    # Add a vertical line at 30 labelled "Latent"
                    fig.add_vline(
                        x=30,
                        line_dash="dash",
                        line_color="gray",
                        annotation_text="Latent",
                        annotation_position="top right",
                        annotation_font_size=10,
                        annotation_font_color="gray",
                    )
                    
                    fig.update_layout(
                        xaxis_title="Recency (Days)",
                        yaxis_title="User Count",
                        coloraxis_colorbar=dict(
                            title="Recency",
                        ),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("Frequency (how often a customer has a session)"):
                    # Make a distribution plot of the frequency, with color gradient green to red
                    frequency_counts = (
                        rfmp_data.groupby("frequency").size().reset_index(name="count")
                    )
                    
                    # Count the number of users with frequency greater than 250
                    greater_than_250 = sum(
                        frequency_counts[frequency_counts["frequency"] > 250]["count"]
                    )
                    less_than_10 = sum(
                        frequency_counts[frequency_counts["frequency"] < 10]["count"]
                    )
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.error(f"**F < 10: {less_than_10:,}**")
                    
                    with col2:
                        st.success(f"**F > 250: {greater_than_250:,}**")
                    
                    # Filter the frequency counts to only include users with frequency less than 200
                    less_than_250 = frequency_counts[frequency_counts["frequency"] < 250]
                    # Create a bar chart of the frequency counts
                    fig = px.bar(
                        less_than_250,
                        x="frequency",
                        y="count",
                        color="frequency",
                        color_continuous_scale="rdylgn",
                    )
                    fig.update_layout(
                        xaxis_title="Frequency (Sessions)",
                        yaxis_title="User Count",
                        coloraxis_colorbar=dict(
                            title="Frequency",
                        ),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("Monetary (how much revenue a customer has generated)"):
                    st.write("***Subscriptions Only***")
                    # Make a distribution plot of the monetary, with color gradient green to red
                    monetary_counts = (
                        rfmp_data.groupby("monetary_value").size().reset_index(name="count")
                    )
                    # Create a bar chart of the monetary counts
                    fig = px.bar(
                        monetary_counts,
                        x="monetary_value",
                        y="count",
                        color="monetary_value",
                        color_continuous_scale="rdylgn",
                    )
                    fig.update_layout(
                        xaxis_title="Monetary (USD)",
                        yaxis_title="User Count",
                        coloraxis_colorbar=dict(
                            title="Monetary",
                        ),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with st.expander(
                    "Points (how many points a customer has/how much activity a customer has)"
                ):
                    # Make a distribution plot of the points, with color gradient green to red
                    points_counts = (
                        rfmp_data.groupby("total_points").size().reset_index(name="count")
                    )
                    # Round points to the nearest 1000
                    points_counts["total_points"] = points_counts["total_points"].round(-3)
                    # Count the number of users with points greater than 50000
                    greater_than_50000 = points_counts[points_counts["total_points"] > 50000]
                    st.write(f"Users with Points > 50,000: {len(greater_than_50000):,}")
                    # Filter the points counts to only include users with points less than 50000
                    less_than_50000 = points_counts[points_counts["total_points"] < 50000]
                    # Create a bar chart of the points counts
                    fig = px.bar(
                        less_than_50000,
                        x="total_points",
                        y="count",
                        color="total_points",
                        color_continuous_scale="rdylgn",
                    )
                    fig.update_layout(
                        xaxis_title="Points",
                        yaxis_title="User Count",
                        coloraxis_colorbar=dict(
                            title="Points",
                        ),
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    # Network Graph Section
    with st.expander("🔗 Network Graph", expanded=False):
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
    