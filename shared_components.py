#!/usr/bin/env python3
"""Shared components for the Streamlit app to reduce code duplication."""

import datetime as dt
import streamlit as st


def initialize_app_state():
    """Initialize the app state with cached data if not already done."""
    if "analytics" not in st.session_state:
        from src.logic_data import MyTeamAnalytics
        from pytz import timezone

        # Get configuration
        server = st.secrets["database"]["server"]
        offset = st.secrets["database"]["offset"]

        # Initialize analytics with cached data
        analytics = MyTeamAnalytics(offset=offset)
        pipeline_data = analytics.run_pipeline()

        # Store in session state
        st.session_state.analytics = analytics
        st.session_state.processed_data = pipeline_data
        st.session_state.team_name = None
        st.session_state.client_id = None
        st.session_state.dfs = initialize_dataframes(pipeline_data)
        st.session_state.refresh_time = dt.datetime.now(timezone(offset))


@st.cache_data(ttl=300)  # Cache dataframe filtering for 5 minutes
def filter_dataframes_by_team(processed_data, team_name):
    """Filter dataframes by team name with caching."""
    dfs = {
        "client": processed_data["clients"],
        "user": processed_data["user_data"],
        "sessions": processed_data["sessions"],
        "engagement": processed_data["engagement"],
        "rewards": processed_data["rewards"],
        "subscriptions": processed_data["subscriptions"],
        "team_data": processed_data["team_data"],
        "rfmp": processed_data["rfmp"],
    }

    if team_name and team_name != "All Teams":
        # Filter dataframes by team_name
        for key in [
            "user",
            "sessions",
            "engagement",
            "rewards",
            "subscriptions",
            "team_data",
        ]:
            if key in dfs and not dfs[key].empty:
                dfs[key] = dfs[key][dfs[key]["team_name"] == team_name]
    else:
        # Create copies to avoid modifying original data
        for key in dfs:
            dfs[key] = dfs[key].copy()

    return dfs


def initialize_dataframes(processed_data, team_name=None):
    """Initialize dataframes with optional team filtering."""
    return filter_dataframes_by_team(processed_data, team_name)


def render_team_selector(key_suffix="", on_change_callback=None):
    """Render a centralized team selector component.

    Args:
        key_suffix (str): Suffix for the selectbox key to make it unique
        on_change_callback (callable): Optional callback when team changes

    Returns:
        str: Selected team name or None for "All Teams"
    """
    # Ensure app state is initialized
    initialize_app_state()

    processed_data = st.session_state.processed_data

    # Get available teams
    all_teams = ["All Teams"] + sorted(
        list(processed_data["clients"]["team_name"].unique())
    )

    # Get current selection index
    current_team = st.session_state.get("team_name", None)
    current_index = (
        0
        if current_team is None
        else (all_teams.index(current_team) if current_team in all_teams else 0)
    )

    # Render selector
    selected_team = st.selectbox(
        "Select a Team",
        options=all_teams,
        index=current_index,
        key=f"team_selector_{key_suffix}",
    )

    # Handle team selection change
    if selected_team != "All Teams":
        if selected_team != st.session_state.get("team_name"):
            # Team changed - update session state
            client_id = processed_data["clients"][
                processed_data["clients"]["team_name"] == selected_team
            ]["_id"].values[0]

            st.session_state.client_id = client_id
            st.session_state.team_name = selected_team
            st.session_state.dfs = filter_dataframes_by_team(
                processed_data, selected_team
            )

            if on_change_callback:
                on_change_callback(selected_team, client_id)

    else:  # All Teams selected
        if st.session_state.get("team_name") is not None:
            # Was previously filtered - reset to all teams
            st.session_state.client_id = None
            st.session_state.team_name = None
            st.session_state.dfs = filter_dataframes_by_team(processed_data, None)

            if on_change_callback:
                on_change_callback(None, None)

    return st.session_state.get("team_name")


def get_current_team_data():
    """Get current team data from session state."""
    initialize_app_state()

    return {
        "analytics": st.session_state.analytics,
        "processed_data": st.session_state.processed_data,
        "dfs": st.session_state.dfs,
        "client_id": st.session_state.get("client_id"),
        "team_name": st.session_state.get("team_name"),
    }


@st.cache_data(ttl=600)  # Cache metrics for 10 minutes
def get_cached_topline_metrics(users_data, engagement_data, subscriptions_data):
    """Get cached topline metrics to avoid recalculation."""
    import pandas as pd

    # Get today and yesterday dates
    today = pd.Timestamp.now().date()
    yesterday = today - pd.Timedelta(days=1)

    # Filter users data
    if not users_data.empty:
        filtered_users = users_data[
            (users_data["user_status"] != "Not Verified")
            & (users_data["user_status"] != "Inactive")
        ]
        total_users = len(filtered_users)

        # New users calculation
        filtered_users["date_joined"] = pd.to_datetime(filtered_users["date_joined"])
        new_users_today = len(
            filtered_users[filtered_users["date_joined"].dt.date == today]
        )
        new_users_yesterday = len(
            filtered_users[filtered_users["date_joined"].dt.date == yesterday]
        )
    else:
        total_users = 0
        new_users_today = 0
        new_users_yesterday = 0

    # Engagement metrics
    if not engagement_data.empty:
        total_engagements = len(engagement_data)
        eng_today = len(engagement_data[engagement_data["datestamp"] == today])
        eng_yesterday = len(engagement_data[engagement_data["datestamp"] == yesterday])

        # Active users
        active_users_today = engagement_data[engagement_data["datestamp"] == today][
            "sender_id"
        ].nunique()
        active_users_yesterday = engagement_data[
            engagement_data["datestamp"] == yesterday
        ]["sender_id"].nunique()
    else:
        total_engagements = 0
        eng_today = 0
        eng_yesterday = 0
        active_users_today = 0
        active_users_yesterday = 0

    # Subscription metrics
    if not subscriptions_data.empty:
        active_subs = len(subscriptions_data[subscriptions_data["is_active"] == True])
    else:
        active_subs = 0

    return {
        "users": {
            "total": total_users,
            "new_today": new_users_today,
            "new_yesterday": new_users_yesterday,
        },
        "engagement": {
            "total": total_engagements,
            "today": eng_today,
            "yesterday": eng_yesterday,
        },
        "active_users": {
            "today": active_users_today,
            "yesterday": active_users_yesterday,
        },
        "subscriptions": {"active": active_subs},
    }


def render_refresh_button():
    """Render a refresh button to clear cache and reload data."""
    if st.button("🔄 Refresh Data"):
        # Clear all caches
        st.cache_data.clear()
        st.cache_resource.clear()

        # Clear session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]

        st.rerun()
