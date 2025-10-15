#!/usr/bin/env python3

import datetime as dt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pytz import timezone
import streamlit as st
import statsmodels.api as sm

pd.options.mode.copy_on_write = True


# define the main user figure
@st.cache_data(ttl=300)  # Cache for 5 minutes
def user_activity_fig(
    client_id, client_df, engagement_df, offset="US/Aleutian", backstop_days=30
):
    """Plot of the number of users per day.

    Returns:
    plt: a plotly object
    """

    today = dt.datetime.now(timezone("UTC"))
    backstop = dt.datetime.now(timezone("UTC")) - pd.Timedelta(days=backstop_days)

    # localize today and yesterday to the offset
    today = pd.to_datetime(today).tz_convert(offset)
    backstop = pd.to_datetime(backstop).tz_convert(offset)

    # reduce to date
    today = today.date()
    backstop = backstop.date()

    # make a dataframe of the days between today and the backstop
    days = pd.date_range(start=backstop, end=today, freq="d")

    # convert created_atAdj to datetime
    engagement_df["created_atAdj"] = pd.to_datetime(engagement_df["created_atAdj"])

    # convert to just date
    engagement_df["created_atAdj"] = engagement_df["created_atAdj"].dt.date

    fig = go.Figure()

    # add a trace for the unique senders per day for each client
    users = (
        engagement_df.groupby(["created_atAdj", "team_name"])
        .sender_username.nunique()
        .unstack()
        .reindex(days)
        .ffill()
        .fillna(0)
    )

    # trim the data to the backstop days
    users = users[-backstop_days:]

    # for each team_name in client_df, plot the user count
    for col in users.columns:
        if col != "total":  # Skip the total column if it exists
            base_color = client_df[client_df["team_name"] == col][
                "client_configuration"
            ].iloc[0]["app"]["secondary_color"]
            color = f"rgba{tuple(int(base_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.4,)}"

            # if there is more than one team, add a trace for the row sum of users across all apps
            if client_id == None:
                # add a trace for the row sum of users across all apps
                users["total"] = users.sum(axis=1)

                fig.add_trace(
                    go.Bar(
                        x=users.index,
                        y=users[col],
                        hovertemplate=f"<b>{col}</b>: %{{y}}</b><extra></extra>",
                        marker_color=color,
                        text=users[col].round(0).astype(int),  # Add data labels
                        textposition="inside",  # Position labels inside bars
                        insidetextanchor="middle",
                        insidetextfont=dict(color="black", size=10),
                        showlegend=False,
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=users.index,
                        y=users["total"],
                        mode="text",
                        customdata=users["total"],
                        hovertemplate="<b>Total</b>: %{customdata} Users<extra></extra>",
                        text=users["total"].round(0).astype(int),
                        textposition="top center",
                        showlegend=False,
                        textfont=dict(size=10),
                    )
                )

            else:
                fig.add_trace(
                    go.Bar(
                        x=users.index,
                        y=users[col],
                        hovertemplate="<b>Users: </b>%{y}<extra></extra>",
                        customdata=users[col],
                        marker_color=color,
                        text=users[col].round(0).astype(int),  # Add data labels
                        textposition="inside",  # Position labels inside bars
                        insidetextanchor="middle",
                        insidetextfont=dict(color="black", size=10),
                        showlegend=False,
                    )
                )

    # change the background to white, remove legend, center the title
    fig.update_layout(
        font=dict(family="Arial, sans-serif`", size=14),
        title=dict(text=f"Daily Active Users ({backstop_days}-Day)"),
        title_font=dict(size=20),
        xaxis_title="Date",
        yaxis_title="Count",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=(
            True if client_id is None else False
        ),  # Show legend only for multiple teams
        hovermode="x unified",
        barmode="stack",
        # Adjust margins to accommodate top labels
        margin_t=50 if client_id is None else 30,
    )

    fig.update_xaxes(
        showgrid=False,
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=False,
        ticks="outside",
    )

    fig.update_yaxes(
        showgrid=False,
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=False,
        ticks="outside",
    )

    return fig


@st.cache_data(ttl=600)  # Cache for 10 minutes
def monthly_active_users_fig(client_id, client_df, engagement_df):
    """Plot of the number of users per month.

    Returns:
    plt: a plotly object
    """

    # convert created_atAdj to datetime
    engagement_df["created_atAdj"] = pd.to_datetime(engagement_df["created_atAdj"])

    # reduce date to the month
    engagement_df["month"] = (
        engagement_df["created_atAdj"].dt.to_period("M").astype(str)
    )

    fig = go.Figure()

    # add a trace for the unique senders per day for each client
    users = (
        engagement_df.groupby(["month", "team_name"])
        .sender_username.nunique()
        .unstack()
        .ffill()
        .fillna(0)
    )

    # for each team_name in client_df, plot the user count
    for col in users.columns:
        if col != "total":  # Skip the total column if it exists
            base_color = client_df[client_df["team_name"] == col][
                "client_configuration"
            ].iloc[0]["app"]["secondary_color"]
            color = f"rgba{tuple(int(base_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.4,)}"

            # if there is more than one team, add a trace for the row sum of users across all apps
            if client_id == None:
                # add a trace for the row sum of users across all apps
                users["total"] = users.sum(axis=1)

                fig.add_trace(
                    go.Bar(
                        x=users.index,
                        y=users[col],
                        hovertemplate=f"<b>{col}</b>: %{{y}}</b><extra></extra>",
                        marker_color=color,
                        text=users[col].round(0).astype(int),  # Add data labels
                        textposition="inside",  # Position labels inside bars
                        insidetextanchor="middle",
                        insidetextfont=dict(color="black", size=10),
                        showlegend=False,
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=users.index,
                        y=users["total"],
                        mode="text",
                        customdata=users["total"],
                        hovertemplate="<b>Total</b>: %{customdata} Users<extra></extra>",
                        text=users["total"].round(0).astype(int),
                        textposition="top center",
                        showlegend=False,
                        textfont=dict(size=10),
                    )
                )

            else:
                fig.add_trace(
                    go.Bar(
                        x=users.index,
                        y=users[col],
                        hovertemplate="<b>Users: </b>%{y}<extra></extra>",
                        customdata=users[col],
                        marker_color=color,
                        text=users[col].round(0).astype(int),  # Add data labels
                        textposition="inside",  # Position labels inside bars
                        insidetextanchor="middle",
                        insidetextfont=dict(color="black", size=10),
                        showlegend=False,
                    )
                )

    # change the background to white, remove legend, center the title
    fig.update_layout(
        font=dict(family="Arial, sans-serif`", size=14),
        title=dict(text="Monthly Active Users"),
        title_font=dict(size=20),
        xaxis_title="Date",
        yaxis_title="Count",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=(
            True if client_id is None else False
        ),  # Show legend only for multiple teams
        hovermode="x unified",
        barmode="stack",
        # Adjust margins to accommodate top labels
        margin_t=50 if client_id is None else 30,
    )

    fig.update_xaxes(
        showgrid=False,
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=False,
        ticks="outside",
        tickformat="%b %Y",
        tickvals=users.index,
    )

    fig.update_yaxes(
        showgrid=False,
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=False,
        ticks="outside",
    )

    return fig


# define the main user figure
@st.cache_data(ttl=300)  # Cache for 5 minutes
def user_growth_fig(
    client_id, client_df, user_df, offset="US/Aleutian", backstop_days=30
):
    """Plot of the number of users per day.

    Returns:
    plt: a plotly object
    """

    fig = go.Figure()

    today = dt.datetime.now(timezone("UTC"))
    backstop = dt.datetime.now(timezone("UTC")) - pd.Timedelta(days=backstop_days - 1)

    # localize today and yesterday to the offset
    today = pd.to_datetime(today).tz_convert(offset)
    backstop = pd.to_datetime(backstop).tz_convert(offset)

    # reduce to date
    today = today.date()
    backstop = backstop.date()

    # make a dataframe of the days between today and the backstop
    days = pd.date_range(start=backstop, end=today, freq="D")

    # filter out "Not Verified" and "Inactive" from the user_df
    user_df = user_df[user_df["status"] != "Not Verified"]
    user_df = user_df[user_df["user_status"] != "Inactive"]

    # group by the date and client_app_id and count the number of users as cumulative
    users = (
        user_df.groupby(["status_date", "team_name", "user_status"])
        .size()
        .unstack()
        .cumsum()
    )

    # get total signups
    total_users = (
        user_df.groupby(["date_joined", "team_name"]).size().unstack().cumsum()
    )

    # split the index tuple into two columns
    users = users.reset_index()

    # set date_joined as the index
    users = users.set_index("status_date")

    users["Total"] = total_users

    # Now reindex with the days and fill forward
    users = users.reindex(days).ffill().bfill().fillna(0)

    users["Active"] = users["Total"] - users["Latent"]

    # Get team name once
    team_name = users["team_name"].iloc[0]

    # Plot Latent users first (grey area at bottom)
    if "Latent" in users.columns:
        fig.add_trace(
            go.Scatter(
                x=users.index,
                y=users["Latent"],
                name="Latent",
                line=dict(color="grey", width=1),
                mode="lines",
                stackgroup="one",
                groupnorm=None,
            )
        )

    # Plot active users once
    if "Active" in users.columns:
        base_color = client_df[client_df["team_name"] == team_name][
            "client_configuration"
        ].iloc[0]["app"]["secondary_color"]
        color = f"rgba{tuple(int(base_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.4,)}"

        fig.add_trace(
            go.Scatter(
                x=users.index,
                y=users["Active"],
                mode="lines",
                name="Active",
                line=dict(width=1, color=color),
                stackgroup="one",
                groupnorm=None,
            )
        )

    if client_id == None:
        # add a trace for the total users per day by summing the active and inactive columns in users

        fig.add_trace(
            go.Scatter(
                x=users.index,
                y=users["Total"],
                mode="lines",
                name="Total Users",
                line=dict(width=2, color="black"),
            )
        )

    # Update layout
    fig.update_layout(
        font=dict(family="Arial, sans-serif", size=14),
        title=dict(text=f"User Counts ({backstop_days}-Day)"),
        title_font=dict(size=20),
        xaxis_title="Date",
        yaxis_title="Count",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
        ),
        hovermode="x unified",
    )

    fig.update_xaxes(
        showgrid=False,
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=False,
        ticks="outside",
    )

    fig.update_yaxes(
        showgrid=False,
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=False,
        ticks="outside",
    )

    return fig


# define the main user figure
@st.cache_data(ttl=300)  # Cache for 5 minutes
def engage_activity_fig(
    client_id, client_df, engagement_df, offset="US/Aleutian", backstop_days=30
):
    """Plot of engagement activity showing daily cumulative sums and hourly totals.

    Returns:
    plt: a plotly object
    """
    engagements = engagement_df.copy()

    today = dt.datetime.now(timezone("UTC"))
    backstop = dt.datetime.now(timezone("UTC")) - pd.Timedelta(days=backstop_days)

    # localize today and yesterday to the offset
    today = pd.to_datetime(today).tz_convert(offset)
    backstop = pd.to_datetime(backstop).tz_convert(offset)

    # reduce to date
    today = today.date()
    backstop = backstop.date()

    # make a dataframe of the days between today and the backstop
    # Use tz_localize with nonexistent='shift_forward' to handle DST transitions
    days = pd.date_range(start=backstop, end=today, freq="h").floor("h")
    try:
        days = days.tz_localize(offset, nonexistent="shift_forward")
    except:
        # If that fails, try using UTC and then converting
        days = days.tz_localize("UTC").tz_convert(offset)

    # localize the days to the offset
    days = days.tz_convert(offset)

    # convert created_atAdj to datetime
    engagements["created_at"] = pd.to_datetime(engagements["created_at"])
    # convert to the hour
    engagements["created_at"] = engagements["created_at"].dt.floor("h")

    # add a trace for the number of engagements per hour for each client
    hourly = (
        engagements.groupby(["created_at", "team_name"])
        .sender_id.count()
        .unstack()
        .reindex(days)
        .fillna(0)
    )
    # change the index to a datetime
    hourly.index = pd.to_datetime(hourly.index)

    # # convert to eastern time
    hourly.index = hourly.index.tz_convert(offset)

    # add a daily cumulative sum of the hourly users
    daily_cumsum = hourly.groupby(hourly.index.date).cumsum()
    daily_cumsum.index = hourly.index  # Restore datetime index

    # trim the data to the backstop days
    hourly = hourly[-backstop_days * 24 :]
    daily_cumsum = daily_cumsum[-backstop_days * 24 :]

    fig = go.Figure()

    # Plot daily cumulative sum for each team
    for col in daily_cumsum.columns:
        base_color = client_df[client_df["team_name"] == col][
            "client_configuration"
        ].iloc[0]["app"]["secondary_color"]
        color = f"rgba{tuple(int(base_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.4,)}"

        fig.add_trace(
            go.Scatter(
                x=daily_cumsum.index,
                y=daily_cumsum[col],
                mode="lines",
                name=f"{col} (daily cumulative)",
                line=dict(width=1, color=color),
                fillcolor=color,
                fill="tonexty",
                stackgroup="one",
            )
        )

        # Add hourly totals as lines
        fig.add_trace(
            go.Scatter(
                x=hourly.index,
                y=hourly[col],
                mode="lines",
                name=f"{col} (hourly)",
                line=dict(width=0.5, color="black"),
                showlegend=True,
            )
        )

    # If there is more than one team, add traces for total engagements
    if client_id is None:
        # Total daily cumulative sum
        daily_cumsum["total"] = daily_cumsum.sum(axis=1)
        fig.add_trace(
            go.Scatter(
                x=daily_cumsum.index,
                y=daily_cumsum["total"],
                mode="lines",
                name="Total (daily cumulative)",
                line=dict(width=2, color="black"),
            )
        )

        # Total hourly
        hourly["total"] = hourly.sum(axis=1)
        fig.add_trace(
            go.Scatter(
                x=hourly.index,
                y=hourly["total"],
                mode="lines",
                name="Total (hourly)",
                line=dict(width=1, color="black", dash="dot"),
            )
        )

    # Update layout
    fig.update_layout(
        font=dict(family="Arial, sans-serif", size=14),
        title=dict(text=f"Engagement Activity ({backstop_days}-Day)"),
        title_font=dict(size=20),
        xaxis_title="Date",
        yaxis_title="Count",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=True,
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor="white",
        ),
        hovermode="x unified",
    )

    fig.update_xaxes(
        showgrid=False,
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=False,
        ticks="outside",
    )

    fig.update_yaxes(
        showgrid=False,
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=False,
        ticks="outside",
    )

    return fig


@st.cache_data(ttl=600)  # Cache for 10 minutes
def churn_figs(cohorts_df, retention_df, time_type="daily", backstop_days=45):
    """A function to generate a plotly figure of the retention curves

    Args:
    cohorts: pandas dataframe

    Returns:
    fig: plotly figure"""
    # numpy and plotly.graph_objects already imported at the top

    # Initialize baseline_lowess with default values
    baseline_lowess = np.column_stack((np.array([]), np.array([])))

    today = dt.datetime.now(timezone("UTC"))

    if time_type == "daily":
        # baseline retentions (convert to percentages)
        baseline = [
            100, 25, 22, 19, 16, 15, 14, 13, 12, 11,
            10, 9.9, 9.8, 9.7, 9.6, 9.5, 9.4, 9.3, 9.2, 9.1,
            9.0, 8.9, 8.8, 8.7, 8.6, 8.5, 8.4, 8.3, 8.2, 8.1,
        ]
        # Ensure baseline matches the number of periods (columns)
        num_periods = (
            cohorts_df.shape[1] if hasattr(cohorts_df, "shape") and len(cohorts_df.shape) > 1 else len(baseline)
        )
        if num_periods > 30:
            for _ in range(30, num_periods):
                baseline.append(8.0)
        elif num_periods < 30:
            baseline = baseline[:num_periods]

    # Use number of periods (columns) for x-axis positions
    num_periods = cohorts_df.shape[1] if hasattr(cohorts_df, "shape") and len(cohorts_df.shape) > 1 else 0
    index = list(range(num_periods))

    # make a list to hold the retention rates
    retention_rates = []

    # Check if we have any data
    if cohorts_df.empty or retention_df.empty:
        # Return empty figure with appropriate message
        fig = go.Figure()
        fig.update_layout(
            title=f"No {time_type} retention data available",
            showlegend=False,
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        return fig

    # Compute retention across periods using top rows (avoid negative slicing)
    num_rows = cohorts_df.shape[0]
    for i in range(num_periods):
        rows_to_include = max(num_rows - i, 0)
        numerator = cohorts_df.iloc[:rows_to_include, i].sum()
        denominator = cohorts_df.iloc[:rows_to_include, 0].sum()

        # check if denominator is 0
        if denominator == 0:
            retention = np.nan
        else:
            retention = (numerator / denominator) * 100  # Convert to percentage

        # add the retention rate to the list
        retention_rates.append(retention)

    # Ensure we have at least one retention rate
    if not retention_rates:
        retention_rates = [100]  # Default to 100% if no data

    # change the first value to 100%
    retention_rates[0] = 100

    # make a lowess of the retention rates and baseline
    lowess = sm.nonparametric.lowess

    # Ensure retention_rates and index have the same length
    min_length = min(len(retention_rates), len(index))
    retention_rates = retention_rates[:min_length]
    index = index[:min_length]

    # Only calculate LOWESS if we have at least 3 data points
    if len(retention_rates) >= 3:
        # fit the lowess to the retention rates
        retention_lowess = lowess(retention_rates, index, frac=0.15)
        retention_lowess[0, 1] = 100  # Set first y-value to 100%

        if time_type == "daily":
            # Align baseline length to min_length and fit LOWESS
            # Recompute baseline to match number of periods
            desired_len = min_length
            if len(baseline) < desired_len:
                baseline = baseline + [8.0] * (desired_len - len(baseline))
            else:
                baseline = baseline[:desired_len]
            baseline_lowess = lowess(baseline, index, frac=0.15)
    else:
        # If not enough points, use raw data for the average line
        retention_lowess = np.column_stack((index, retention_rates))
        if time_type == "daily":
            baseline_lowess = np.column_stack((index, baseline[:min_length]))

    # make a plotly figure
    fig = go.Figure()

    # for each column of retention_df add a trace to the figure
    for i in range(retention_df.shape[0]):
        # get the date of join for cohort name
        cohort_date = cohorts_df.index[i]
        if time_type == "daily":
            # convert the date to a datetime object
            cohort_date = pd.to_datetime(cohort_date)
            # if the date is today, set the name to "Today"
            if cohort_date == today:
                cohort_date = "Today"
            else:
                # format the date as "DD MMM"
                cohort_date = cohort_date.strftime("%d %b")
        elif time_type == "weekly":
            # format the date as "Week DD"
            cohort_date = f"Week {cohort_date[-2:]}"
        else:
            # format the date as "Month DD"
            cohort_date = f"Month {cohort_date[-2:]}"

        # Convert values to percentages
        retention_values = retention_df.iloc[i, :min_length] * 100
        # add the retention rates to the figure
        fig.add_trace(
            go.Scatter(
                x=index,
                y=retention_values,
                line=dict(color="lightgray", width=1),
                mode="lines",
                name=f"Cohort {cohort_date}",
                hovertemplate="%{y}%",
            )
        )

    # add the retention rates to the figure
    fig.add_trace(
        go.Scatter(
            x=index,
            y=retention_lowess[:, 1],
            name="Avg Retention Rate",
            line=dict(color="green", width=2),
            mode="lines",
            hovertemplate="%{y}%",
        )
    )

    # Add baseline retention rates for daily view
    if (
        time_type == "daily"
        and baseline_lowess is not None
        and hasattr(baseline_lowess, "shape")
        and baseline_lowess.size > 0
    ):
        fig.add_trace(
            go.Scatter(
                x=index,
                y=(
                    baseline_lowess[:, 1]
                    if len(baseline_lowess.shape) > 1 and baseline_lowess.shape[1] > 1
                    else np.zeros_like(index)
                ),
                name="Industry Rate",
                line=dict(color="red", width=2, dash="dash"),
                mode="lines",
                hovertemplate="%{y}%",
            )
        )

    title = (
        "30-Day"
        if time_type == "daily"
        else "12-Week" if time_type == "weekly" else "12-Month"
    )
    period_label = (
        "Days"
        if time_type == "daily"
        else "Weeks" if time_type == "weekly" else "Months"
    )
    max_x = 30 if time_type == "daily" else 12 if time_type == "weekly" else 12

    fig.update_layout(
        font=dict(family="Arial, sans-serif", size=14),
        title=dict(
            text=f"{title} Retention Rates (Last {backstop_days}-Day New Users)"
        ),
        title_font=dict(size=20),
        xaxis_title=period_label,
        yaxis_title="Retention Rate (%)",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=False,
        hovermode="closest",
        legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="right", x=1),
    )

    fig.update_xaxes(
        showgrid=False,
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=False,
        ticks="outside",
        range=[0, max_x],  # Set range based on time_type
        tickformat="d",  # Use integer format
        dtick=1,  # Force integer ticks
    )

    fig.update_yaxes(
        showgrid=False,
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=False,
        ticks="outside",
        tickformat=".0f",  # No decimal places
        ticksuffix="%",  # Add % symbol
        range=[0, 100],  # Set range from 0% to 100%
    )

    return fig


def create_retention_heatmap(self, period_type="weekly"):
    """Create retention heatmap visualization"""
    retention_data = self._processed_data["retention"]
    rates = retention_data[f"{period_type.lower()}"]["rates"]

    if period_type == "weekly":
        # Convert week numbers to start dates
        new_index = []
        for week in rates.index:
            year, week_num = week.split("-")
            # Create date from year and week number
            week_start = dt.datetime.strptime(f"{year}-W{week_num}-1", "%Y-W%W-%w")
            # Format as 'YYYY-MM-DD'
            new_index.append(week_start.strftime("%Y-%m-%d"))

        # Update the index
        rates.index = new_index

    fig = go.Figure(
        data=go.Heatmap(
            z=rates.values,
            x=rates.columns,
            y=rates.index,
            colorscale="RdYlGn",
            zmin=0,
            zmax=100,
        )
    )

    period_label = "Weeks" if period_type == "weekly" else "Days"
    fig.update_layout(
        title=f"Cohort Retention Heatmap ({period_label})",
        xaxis_title=f"{period_label} Since First Activity",
        yaxis_title="Cohort Start Date",
        template="plotly_white",
    )

    return fig


def referral_time_fig(
    client_id,
    client_df,
    users_df,
    offset="US/Aleutian",
    backstop_days=90,
):
    """Create a time series figure of the number of referrals over time.

    Args:
    client: str
    users_df: pandas dataframe
    offset: str

    Returns:
    fig: plotly figure"""

    fig = go.Figure()

    today = dt.datetime.now(timezone("UTC"))
    backstop = dt.datetime.now(timezone("UTC")) - pd.Timedelta(days=120)

    # localize today and yesterday to the offset
    today = pd.to_datetime(today).tz_convert(offset)
    backstop = pd.to_datetime(backstop).tz_convert(offset)

    # reduce to date
    today = today.date()
    backstop = backstop.date()

    # make a dataframe of the days between today and the backstop
    days = pd.date_range(start=backstop, end=today, freq="D")

    # filter the users_df to records where "referred_by" != None
    users_df = users_df[users_df["referred_by"].notnull()]

    # add a trace for the number of users created each day by client_id
    users = (
        users_df.groupby(["date_joined", "team_name"])
        .size()
        .unstack()
        .cumsum()
        .reindex(days)
        .ffill()
        .fillna(0)
    )

    # trim the data to the backstop days
    users = users[-backstop_days:]

    # for each team_name in client_df, plot the user count
    for col in users.columns:
        base_color = client_df[client_df["team_name"] == col][
            "client_configuration"
        ].iloc[0]["app"]["secondary_color"]
        color = f"rgba{tuple(int(base_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.4,)}"
        fig.add_trace(
            go.Scatter(
                x=users.index,
                y=users[col],
                mode="lines",
                name=client_df[client_df["team_name"] == col]["team_name"].values[0],
                line=dict(
                    width=1,
                    color=color,
                ),
                fillcolor=color,
                fill="tonexty",
                stackgroup="one",
            )
        )

    if client_id == None:

        # add a trace for the row sum of users across all apps
        users["total"] = users.sum(axis=1)

        fig.add_trace(
            go.Scatter(
                x=users.index,
                y=users["total"],
                mode="lines",
                name="Total",
                line=dict(width=2, color="black"),
            )
        )

    # change the background to white, remove legend, center the title
    fig.update_layout(
        font=dict(family="Arial, sans-serif`", size=14),
        title=dict(text="Referral Growth"),
        xaxis_title="Date",
        yaxis_title="Count",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=True,
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor="white",
        ),
        hovermode="x unified",
    )

    fig.update_xaxes(
        showgrid=False,
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=False,
        ticks="outside",
    )

    fig.update_yaxes(
        showgrid=False,
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=False,
        ticks="outside",
    )

    return fig


def referral_bar_chart(
    client_id,
    client_df,
    users_df,
    offset="US/Aleutian",
    backstop_days=90,
):
    """Create a bar chart of the top referring users.

    Args:
    client: str
    users_df: pandas dataframe
    offset: str

    Returns:
    fig: plotly figure"""

    fig = go.Figure()

    # today
    backstop = dt.datetime.now(timezone("UTC")) - pd.Timedelta(days=backstop_days)

    # localize to the offset
    backstop = pd.to_datetime(backstop).tz_convert(offset)

    # reduce to date
    backstop = backstop.date()

    # if the client_id is not null, filter the referred users to the client_id
    if client_id != None:
        users_df = users_df[users_df["client_app_id"] == client_id]

    # referred users is users_df referees len >0
    referred_users = users_df[users_df["referees"].apply(len) > 0]

    # keep username, status, client_app_id, and referees columns
    referred_users = referred_users[["username", "client_app_id", "referees"]]

    # explode
    referred_users = referred_users.explode("referees")

    # join with users_df to get the phone_number
    referred_users = referred_users.merge(
        users_df, left_on="referees", right_on="phone_number"
    )

    # keep CreatedAtAdjDate > backstop
    referred_users = referred_users[referred_users["date_joined"] >= backstop]

    # groupby username_x and client_app_id_x and count the number of referees
    referred_users = (
        referred_users.groupby(["username_x", "client_app_id_x"])
        .size()
        .reset_index(name="referrals")
    )

    # sort by referrals
    referred_users = referred_users.sort_values(by="referrals", ascending=False)

    # keep the top 10
    referred_users = referred_users.head(10)

    # join with users_df to get the user_status
    user_status = users_df[["username", "user_status"]]
    referred_users = referred_users.merge(
        user_status, left_on="username_x", right_on="username"
    )

    # for each row, add it to a bar chart and look up the color from the client_df
    for i, row in referred_users.iterrows():
        # get the trace_color_rgb and name from the client_df
        base_color = client_df.loc[
            client_df["_id"] == row["client_app_id_x"], "client_configuration"
        ].iloc[0]["app"]["secondary_color"]
        color = f"rgba{tuple(int(base_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.4,)}"
        # team_name = client_df.loc[
        #     client_df["_id"] == row["client_app_id_x"], "team_name"
        # ].values[0]

        fig.add_trace(
            go.Bar(
                x=[row["referrals"]],
                y=[f"{row['username_x']}<br>({row['user_status']})"],
                name=row["username_x"],
                orientation="h",
                marker=dict(color=color),
                hovertemplate=f"{row['username_x']}<br>Referrals: {row['referrals']}<extra></extra>",
                # add a text label to the bar
                text=[row["referrals"]],
                textposition="inside",
                textfont=dict(color="black"),
            )
        )

    fig.update_layout(
        title="Top Referring Users",
        xaxis_title="Number of Referrals",
        # yaxis_title="Referring User",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=100, r=0, t=50, b=0),
        showlegend=False,
    )

    fig.update_yaxes(autorange="reversed")

    return fig


def raffle_entry_bar_chart(
    client_id,
    client_df,
    raffles_df,
    offset="US/Aleutian",
    backstop_days=90,
):
    """Create a bar chart of the top raffles.

    Args:
    client: str
    raffles_df: pandas dataframe
    offset: str

    Returns:
    fig: plotly figure"""

    # today
    backstop = dt.datetime.now(timezone("UTC")) - pd.Timedelta(days=backstop_days)

    # localize to the offset
    backstop = pd.to_datetime(backstop).tz_convert(offset)

    # reduce to date
    backstop = backstop.date()

    fig = go.Figure()

    # if the client_id is not null, filter the referred users to the client_id
    if client_id != None:
        raffles_df = raffles_df[raffles_df["client_app_id"] == client_id]

    # convert close_date to a datetime object
    raffles_df["close_date"] = pd.to_datetime(raffles_df["close_date"])

    # convert close_date to the offset
    raffles_df["close_date"] = raffles_df["close_date"].dt.tz_convert(offset)

    # convert close_date to a date object
    raffles_df["close_date"] = raffles_df["close_date"].dt.date

    # filter the raffles to the backstop
    raffles_df = raffles_df[raffles_df["close_date"] >= backstop]

    # sort by entries
    raffles_df = raffles_df.sort_values("total_no_of_entries", ascending=False)

    # keep the top 10
    raffles_df = raffles_df.head(10)

    # for each row, add it to a bar chart and look up the color from the client_df
    for i, row in raffles_df.iterrows():
        if row["is_premium"] == True:
            prem_flag = "Premium"
        else:
            prem_flag = "Free"
        # get the trace_color_rgb and name from the client_df
        base_color = client_df.loc[
            client_df["_id"] == row["client_app_id"], "client_configuration"
        ].iloc[0]["app"]["secondary_color"]
        color = f"rgba{tuple(int(base_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.4,)}"
        team_name = client_df.loc[
            client_df["_id"] == row["client_app_id"], "team_name"
        ].values[0]

        # for each trace make the y the title and CloseDateAdjDate combined
        fig.add_trace(
            go.Bar(
                x=[row["total_no_of_entries"]],
                y=[f"{row['title']} <br> {prem_flag} ({row['close_date']})"],
                orientation="h",
                marker=dict(color=color),
                hovertemplate=f"{team_name}<br>Entries: {row['total_no_of_entries']}<extra></extra>",
                text=row["total_no_of_entries"],
                textposition="inside",
                textfont=dict(color="black"),
            )
        )

    fig.update_layout(
        title="Top Raffles",
        xaxis_title="Number of Entries",
        # yaxis_title=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=200, r=0, t=50, b=0),
        showlegend=False,
    )

    fig.update_yaxes(autorange="reversed")

    return fig


def raffle_entry_time_chart(
    client_id,
    client_df,
    raffles_df,
    offset="US/Aleutian",
    backstop_days=90,
):
    """Create a bar chart of the top raffles.

    Args:
    client: str
    raffles_df: pandas dataframe
    offset: str

    Returns:
    fig: plotly figure"""

    today = dt.datetime.now(timezone("UTC"))
    backstop = dt.datetime.now(timezone("UTC")) - pd.Timedelta(days=backstop_days)

    # localize to the offset
    today = pd.to_datetime(today).tz_convert(offset)
    backstop = pd.to_datetime(backstop).tz_convert(offset)

    # reduce to date
    today = today.date()
    backstop = backstop.date()

    fig = go.Figure()

    # if the client_id is not null, filter the referred users to the client_id
    if client_id != None:
        raffles_df = raffles_df[raffles_df["client_app_id"] == client_id]

    # days between today and backstop
    days = days = pd.date_range(start=backstop, end=today, freq="D")

    # convert close_date to a datetime object
    raffles_df["close_date"] = pd.to_datetime(raffles_df["close_date"])

    # convert close_date to the offset
    raffles_df["close_date"] = raffles_df["close_date"].dt.tz_convert(offset)

    # convert close_date to a date object
    raffles_df["close_date"] = raffles_df["close_date"].dt.date

    # group by the date and client_app_id and count the total_num_of_entries as cumulative
    raffle_entries = (
        raffles_df.groupby(["close_date", "client_app_id"])
        .agg({"total_no_of_entries": "sum"})
        .unstack()
        .cumsum()
        .reindex(days)
        .ffill()
        .fillna(0)
    )

    # remove the top level of the column index
    raffle_entries.columns = raffle_entries.columns.droplevel(0)

    # trim the data to the backstop days
    raffle_entries = raffle_entries[-backstop_days:]

    # for each team_name in client_df, plot the user count
    for col in raffle_entries.columns:
        base_color = client_df.loc[
            client_df["_id"] == col, "client_configuration"
        ].iloc[0]["app"]["secondary_color"]
        color = f"rgba{tuple(int(base_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.4,)}"
        fig.add_trace(
            go.Scatter(
                x=raffle_entries.index,
                y=raffle_entries[col],
                mode="lines",
                name=client_df[client_df["_id"] == col]["team_name"].values[0],
                line=dict(
                    width=1,
                    color=color,
                ),
                fillcolor=color,
                fill="tonexty",
                stackgroup="one",
            )
        )

    if client_id == None:

        # add a trace for the total users per day
        raffle_entries = (
            raffles_df.groupby("close_date")
            .agg({"total_no_of_entries": "sum"})
            .cumsum()
            .reindex(days)
            .ffill()
            .fillna(0)
        )

        # conver to a series
        raffle_entries = raffle_entries.squeeze()

        # trim the data to the backstop days
        raffle_entries = raffle_entries[-backstop_days:]

        fig.add_trace(
            go.Scatter(
                x=raffle_entries.index,
                y=raffle_entries,
                mode="lines",
                name="Total",
                line=dict(width=2, color="black"),
            )
        )

    # change the background to white, remove legend, center the title
    fig.update_layout(
        font=dict(family="Arial, sans-serif`", size=14),
        title=dict(text=f"Raffle Activity ({backstop_days}-Day)"),
        title_font=dict(size=20),
        xaxis_title="Date",
        yaxis_title="Entries",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=False,
        # legend=dict(
        #     x=0.01,
        #     y=0.99,
        #     bgcolor="white",
        # ),
        hovermode="x unified",
    )

    fig.update_xaxes(
        showgrid=False,
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=False,
        ticks="outside",
    )

    fig.update_yaxes(
        showgrid=False,
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=False,
        ticks="outside",
    )

    return fig


@st.cache_data(ttl=300)  # Cache for 5 minutes
def dau_line_chart(engagement_df, backstop_days=30):
    """Create a line chart showing DAU with 7-day and 30-day rolling averages.

    Args:
        engagement_df (pd.DataFrame): DataFrame containing engagement data
        offset (str): Timezone offset
        backstop_days (int): Number of days to look back

    Returns:
        go.Figure: Plotly figure object
    """

    # reduce to just the sender_id and datestamp
    user_activity = engagement_df[["sender_id", "datestamp"]]

    # drop duplicates
    user_activity = user_activity.drop_duplicates()

    # First, set datestamp to the index
    user_activity = user_activity.set_index("datestamp")
    user_activity = user_activity.sort_index()

    # Create a function to count unique users in a time window by using days to set the range
    def count_unique_users_in_window(activity_df, day_window, col_name="users"):
        """Count unique users in a rolling time window.

        Args:
            activity_df (pd.DataFrame): DataFrame containing engagement data
            day_window (int): Number of days in the window
            col_name (str): Name for the output series

        Returns:
            pd.Series: Series with counts of unique users for each window
        """

        # Initialize counts dictionary
        counts = {}

        # For each day in the index
        for day in activity_df.index:
            # Calculate window start date
            window_start = day - pd.Timedelta(days=day_window - 1)

            # Filter engagement data to this time window
            mask = (activity_df.index >= window_start) & (activity_df.index <= day)
            window_data = activity_df[mask]

            # Count unique users in this window
            counts[day] = window_data["sender_id"].nunique()

        # Return a series with the counts, using the original index
        return pd.Series(counts, index=activity_df.index, name=col_name)

    # Then calculate the rolling windows
    user_activity["rolling_1d"] = count_unique_users_in_window(user_activity, 1)
    user_activity["rolling_7d"] = count_unique_users_in_window(user_activity, 7)
    user_activity["rolling_30d"] = count_unique_users_in_window(user_activity, 30)

    # drop sender_id column
    user_activity = user_activity.drop(columns=["sender_id"])

    # drop duplicates
    user_activity = user_activity.drop_duplicates()

    # set datestamp as index
    user_activity = user_activity.sort_index()

    # fill the missing values with 0
    user_activity = user_activity.fillna(0)

    # calculate the backstop date
    backstop_date = user_activity.index[-1] - pd.Timedelta(days=backstop_days)

    # trim the data to the backstop days
    user_activity = user_activity[user_activity.index >= backstop_date]

    # drop the last row
    user_activity = user_activity.iloc[:-1]

    # Create figure
    fig = go.Figure()

    # Add traces
    fig.add_trace(
        go.Scatter(
            x=user_activity.index,
            y=user_activity["rolling_1d"],
            name="Daily Active Users",
            line=dict(color="black", width=2),
            mode="lines",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=user_activity.index,
            y=user_activity["rolling_7d"],
            name="7-Day Active Users",
            line=dict(color="darkgray", width=1, dash="dash"),
            mode="lines",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=user_activity.index,
            y=user_activity["rolling_30d"],
            name="30-Day Active Users",
            line=dict(color="darkgray", width=1, dash="dot"),
            mode="lines",
        )
    )

    # Update layout
    fig.update_layout(
        title="User Activity Windows",
        xaxis_title="Date",
        yaxis_title="Number of Users",
        hovermode="x unified",
        showlegend=True,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=0, r=0, t=50, b=0),
    )

    # Update axes
    fig.update_xaxes(
        showgrid=False,
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=False,
        ticks="outside",
    )

    fig.update_yaxes(
        showgrid=False,
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=False,
        ticks="outside",
    )

    return fig, user_activity


# # Define rfp_level function
# def rfp_level(df):
#     if df["RFP_Score"] >= 9:
#         return "Top"
#     elif (df["RFP_Score"] >= 5) and (df["RFP_Score"] < 9):
#         return "Middle"
#     else:
#         return "Low"


# def retention_analysis(
#     client,
#     users_df,
#     rewards_df,
# ):
#     """Main runner function for the retention analysis

#     Args:
#     client: A pymongo client object
#     users_df: A dataframe of users
#     rewards_df: A dataframe of rewards

#     Returns:
#     daily_retention: A series of the daily retention rate for 30 days
#     weekly_retention: A series of the weekly retention rate for 26 weeks
#     user_df_rtp: A dataframe of users with the recency, frequency, and reward points columns added and a latency or churned flag
#     """

#     if client != None:

#         # filter the rewards_df to only include the client_id
#         rewards_df = rewards_df[rewards_df["client_app_id"] == client]

#     # make sure the CreatedAtAdjDate column is a datetime object
#     rewards_df["CreatedAtAdjDate"] = pd.to_datetime(rewards_df["CreatedAtAdjDate"])

#     # create RewardDay columns
#     rewards_df["RewardDay"] = rewards_df["CreatedAtAdjDate"]

#     # Group by sender_id and select the RewardDay value
#     grouping = rewards_df.groupby("sender_id")["RewardDay"]

#     # Assign a minimum RewardDay value to the dataset
#     rewards_df["RewardDay"] = grouping.transform("min")

#     # create a CohortDay column with is the RewardDay
#     rewards_df["CohortDay"] = rewards_df["RewardDay"]

#     # Get the datediff in days from the `CreatedAtAdjDate` column and the `CohortDay` column
#     rewards_df["CohortDayIndex"] = (
#         rewards_df["CreatedAtAdjDate"] - rewards_df["CohortDay"]
#     ).dt.days + 1

#     # create a new column called CohortWeek which is the annual week of the CohortDay
#     rewards_df["CohortWeek"] = rewards_df["CohortDay"].dt.strftime("%Y-%U")

#     # define a new column for RewardWeek the same way
#     rewards_df["RewardWeek"] = rewards_df["CreatedAtAdj"].dt.strftime("%Y-%U")

#     # get the difference in years between the CohortWeek and RewardWeek
#     rewards_df["CohortWeekIndex"] = (
#         (
#             (rewards_df["RewardWeek"].str[:4].astype(int))
#             - rewards_df["CohortWeek"].str[:4].astype(int)
#         )
#         * 52
#         + (
#             rewards_df["RewardWeek"].str[5:].astype(int)
#             - rewards_df["CohortWeek"].str[5:].astype(int)
#         )
#         + 1
#     )

#     rewards_df_rfp = rewards_df.copy()

#     snapshot_date = max(rewards_df_rfp.CreatedAtAdj) + dt.timedelta(days=1)

#     # Calculate Recency, Frequency and Monetary value for each customer
#     rewards_df_rfp = rewards_df_rfp.groupby(["sender_id"]).agg(
#         {
#             "CreatedAtAdj": lambda x: (snapshot_date - x.max()).days,
#             "_id": "count",
#             "points": "sum",
#         }
#     )

#     # Rename the columns
#     rewards_df_rfp.rename(
#         columns={"CreatedAtAdj": "Recency", "_id": "Frequency", "points": "PointValue"},
#         inplace=True,
#     )

#     # Create labels for Recency and Frequency
#     r_labels = list(range(4, 0, -1))

#     # Assign these labels to three equal percentile groups
#     r_groups = pd.qcut(
#         rewards_df_rfp["Recency"], q=4, labels=r_labels, duplicates="drop"
#     )

#     # Assign these labels to three equal percentile groups
#     f_groups = pd.qcut(rewards_df_rfp["Frequency"], q=4, labels=range(1, 5))

#     # Create new columns R and F
#     rewards_df_rfp = rewards_df_rfp.assign(R=r_groups.values, F=f_groups.values)

#     # Assign these labels to three equal percentile groups
#     p_groups = pd.qcut(rewards_df_rfp["PointValue"], q=4, labels=range(1, 5))

#     # Create new column P
#     rewards_df_rfp = rewards_df_rfp.assign(P=p_groups.values)

#     # Calculate RFP_Score
#     rewards_df_rfp["RFP_Score"] = rewards_df_rfp[["R", "F", "P"]].sum(axis=1)

#     # RFP_Segment is the combined string of RFP values
#     rewards_df_rfp["RFP_Segment"] = (
#         rewards_df_rfp["R"].astype(str)
#         + rewards_df_rfp["F"].astype(str)
#         + rewards_df_rfp["P"].astype(str)
#     )

#     # Create a new variable RFP_Level
#     rewards_df_rfp["RFP_Level"] = rewards_df_rfp.apply(rfp_level, axis=1)

#     # # subset rewards_df_rtp to only include the sender_id, RFP_Level, RFP_Score, Recency, Frequency, and PointValue
#     # rewards_df_rtp_sub = rewards_df_rtp[["RFP_Level", "RFP_Score", "Recency", "Frequency", "PointValue"]].copy()
#     rewards_df_rfp = rewards_df_rfp[
#         ["Recency", "Frequency", "PointValue", "RFP_Level", "RFP_Score"]
#     ].copy()

#     # for the sender_id, merge the recency, frequency, and pointvalue to user_df
#     user_df_rfp = users_df.merge(
#         rewards_df_rfp, left_on="_id", right_on="sender_id", how="left"
#     )

#     # where the Recency is null, change the RFP_Level to "FailLaunch"
#     user_df_rfp.loc[user_df_rfp.Recency.isnull(), "RFP_Level"] = "FailLaunch"

#     # where the recency is null, fill with the maximum recency
#     user_df_rfp.fillna({"Recency": user_df_rfp.Recency.max()}, inplace=True)

#     # where the frequency is null, fill with 0
#     user_df_rfp.fillna({"Frequency": 0}, inplace=True)

#     # where the pointvalue is null, fill with 0
#     user_df_rfp.fillna({"PointValue": 0}, inplace=True)

#     # where the RFP_Level is null, fill with "N/A"
#     user_df_rfp.fillna({"RFP_Level": "N/A"}, inplace=True)

#     # where the RFP_Score is null, fill with 0
#     user_df_rfp.fillna({"RFP_Score": 0}, inplace=True)

#     # where the Recency is >= 14, create a flag called High_Prob_Churn
#     user_df_rfp["High_Prob_Off"] = (
#         (user_df_rfp.Recency >= 14) & (user_df_rfp.Recency < 21)
#     ).astype(int)

#     # where the Recency is >= 5 and < 14, create a flag called High_Prob_Latent
#     user_df_rfp["High_Prob_Latent"] = (
#         (user_df_rfp.Recency > 5) & (user_df_rfp.Recency < 14)
#     ).astype(int)

#     # where the Recency is > 21, create a flag called High_Prob_Churn
#     user_df_rfp["High_Prob_Churn"] = (user_df_rfp.Recency >= 21).astype(int)

#     # where the Recency is < 5, create a flag called High_Prob_Active
#     user_df_rfp["High_Prob_Active"] = (user_df_rfp.Recency <= 5).astype(int)

#     return rewards_df, user_df_rfp


# def cohort_counts(rewards_df, time_type, offset="US/Aleutian", backstop_days=45):
#     """Create a cohort count dataframe.

#     Args:
#     rewards_df: pandas dataframe

#     Returns:
#     cohort_counts: pandas dataframe"""

#     if time_type == "daily":
#         counter = "CohortDay"
#         indexer = "CohortDayIndex"
#     else:
#         counter = "CohortWeek"
#         indexer = "CohortWeekIndex"

#     # get the date from time and make a created_atAdj
#     today = dt.datetime.now(timezone("UTC"))

#     # localize today and yesterday to the offset
#     today = pd.to_datetime(today).tz_convert(offset)

#     # reduce to date
#     today = today.date()

#     # set today time to midnight
#     today = dt.datetime.combine(today, dt.datetime.min.time())

#     # Group by `CohortDay`
