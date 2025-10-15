#!/usr/bin/env python3
"""Core analytics functionality for MyTeam data processing."""

import tomllib
import datetime as dt
import certifi
import networkx as nx
import pandas as pd
import streamlit as st

# import statsmodels.api as sm

from pymongo import MongoClient

pd.options.mode.copy_on_write = True


# Cached database connection
@st.cache_resource
def get_database_connection(server, database="Pheonix"):
    """Get cached database connection."""
    client = MongoClient(
        server,
        tlsCAFile=certifi.where(),
        tz_aware=True,
        datetime_conversion="DATETIME_AUTO",
    )
    return getattr(client, database)


# Cached data fetching functions
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_mongodb_data(server, database="Pheonix"):
    """Fetch all data from MongoDB with caching."""
    db = get_database_connection(server, database)

    collections = {
        "clients": [
            "_id",
            "team_name",
            "client_configuration",
        ],
        "users": [
            "_id",
            "created_at",
            "client_app_id",
            "daily_accumulated_fan_points",
            "referred_by",
            "following",
            "followers",
            "referees",
            "reward_points",
            "total_reward_points",
            "country",
            "dob",
            "email_address",
            "phone_number",
            "username",
            "is_verified",
            "is_deleted",
        ],
        "rewards": [
            "created_at",
            "client_app_id",
            "sender_id",
            "receiver_id",
            "points",
            "type",
        ],
        "posts": [
            "created_at",
            "user_id",
            "client_app_id",
            "liked_users",
            "post_type",
        ],
        "comments": [
            "created_at",
            "user_id",
            "post_id",
            "client_app_id",
            "text",
            "number_of_replies",
            "liked_users",
        ],
        "raffles": [
            "_id",
            "points",
            "title",
            "close_date",
            "start_date",
            "is_premium",
            "client_app_id",
            "status",
            "total_no_of_entries",
        ],
        "subscriptions": [
            "user_id",
            "client_app_id",
            "subscription_start_date",
            "subscription_end_date",
            "is_active",
            "duration",
            "sub_type",
            "platform",
            "amount",
            "payment_status",
        ],
    }

    raw_data = {}
    for collection_name, fields in collections.items():
        collection = getattr(db, collection_name)
        projection = {field: 1 for field in fields}
        raw_data[collection_name] = list(collection.find({}, projection))

    # Convert to DataFrames
    for collection_name in raw_data:
        raw_data[collection_name] = pd.DataFrame(raw_data[collection_name])

    return raw_data


# Cached processing functions for expensive operations
@st.cache_data(ttl=600)  # Cache for 10 minutes
def process_cached_user_data(users_data, _subscriptions_data):
    """Process user data with caching."""
    users = users_data.copy()

    if users.empty:
        return pd.DataFrame()

    # Basic user processing logic (simplified for caching)
    users["status"] = "active"  # This would contain actual logic

    return users


@st.cache_data(ttl=300)  # Cache for 5 minutes
def process_cached_engagement_data(rewards_data, _users_data):
    """Process engagement data with caching."""
    if rewards_data.empty:
        return pd.DataFrame()

    # Basic engagement processing (simplified for caching)
    engagement_df = rewards_data.copy()

    return engagement_df


@st.cache_data(ttl=1200)  # Cache for 20 minutes (expensive)
def process_cached_network_data(users_data, rewards_data):
    """Process network analysis with caching."""
    if users_data.empty or rewards_data.empty:
        return pd.DataFrame()

    # Network processing would go here
    network_df = pd.DataFrame()

    return network_df


@st.cache_data(ttl=600)  # Cache for 10 minutes
def process_cached_retention_data(
    engagement_df, _time_type="weekly", _backstop_days=45
):
    """Process retention analysis with caching."""
    if engagement_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Retention processing would go here
    cohort_counts = pd.DataFrame()
    retention_rates = pd.DataFrame()

    return cohort_counts, retention_rates


class MyTeamAnalytics:
    def __init__(self, database="Pheonix", offset="US/Aleutian", env="local"):
        """
        Initialize the analytics pipeline with database connection.

        Args:
            database (str): Name of the database to connect to
            offset (str): Timezone offset for data processing
            env (str): Environment type ('local' or 'streamlit')
        """
        # Get database connection string based on environment
        if env == "local":
            # Load from local secrets.toml
            with open(".streamlit/secrets.toml", "rb") as f:
                secrets = tomllib.load(f)
            self.server = secrets["database"]["server"]
            self.offset = secrets["database"]["offset"]
        elif env == "streamlit":
            # Load from Streamlit secrets
            self.server = st.secrets["server"]
            self.offset = st.secrets["offset"]
        else:
            raise ValueError("env must be either 'local' or 'streamlit'")

        # Use cached database connection
        self.db = get_database_connection(self.server, database)
        self.offset = offset

        # Initialize storage for processed data
        self.client_id = None
        self.team_name = None
        self._raw_data = {}
        self._processed_data = {}
        self._all_raw_data = {}  # New storage for all data

        # Fetch all data on initialization
        self.fetch_all_data()

    def fetch_all_data(self):
        """Fetch all data from MongoDB without client filtering."""
        # Use cached data fetching
        self._all_raw_data = fetch_mongodb_data(self.server)

        # Copy to _raw_data for backward compatibility
        for collection_name, df in self._all_raw_data.items():
            # Process timestamps if needed
            if "created_at" in df.columns:
                df = self._process_timestamps(df, ["created_at"])
            self._raw_data[collection_name] = df

    def team_name_mapping(self):
        # Create teamname dictionary
        teamname_map = self._raw_data["clients"].set_index("_id")["team_name"].to_dict()
        return teamname_map

    def _process_timestamps(self, df, date_columns):
        """
        Process timestamps to add UTC and adjusted time columns.

        Args:
            df (DataFrame): DataFrame containing timestamp columns
            date_columns (list): List of column names containing timestamps

        Returns:
            DataFrame with processed timestamp columns
        """
        for col in date_columns:
            # Convert to pandas datetime while preserving timezone
            df[col] = pd.to_datetime(df[col], utc=True)

            # Create UTC version
            df[f"{col}UTC"] = df[col]

            # Convert to specified timezone
            df[f"{col}Adj"] = df[col].dt.tz_convert(self.offset)

            # Create date version
            df[f"{col}AdjDate"] = df[f"{col}Adj"].dt.date

        return df

    def process_client_data(self):
        """Process client-level metrics."""
        clients = self._raw_data["clients"].copy()

        # remove where team_name is None
        clients = clients[clients["team_name"].notna()]

        self._processed_data["clients"] = clients

    def process_user_data(self):
        """Process user-level metrics."""
        users = self._raw_data["users"].copy()

        # add a status column to combine is_deleted and is_verified
        for index, row in users.iterrows():
            if row["is_deleted"]:
                users.at[index, "status"] = "Deleted"
            elif row["is_verified"]:
                users.at[index, "status"] = "Active"
            else:
                users.at[index, "status"] = "Not Verified"

        # Extract country name from country JSON
        users["country"] = users["country"].apply(
            lambda x: x.get("name") if isinstance(x, dict) else None
        )

        # map the team name to the client_app_id
        users["team_name"] = users["client_app_id"].map(self.team_name_mapping())

        # Calculate key metrics
        metrics = pd.DataFrame(
            {
                "username": users["username"],
                "status": users["status"],
                "team_name": users["team_name"],
                "country": users["country"],
                "dob": users["dob"],
                "email": users["email_address"],
                "phone_number": users["phone_number"],
                "date_joined": users["created_atAdjDate"],
                "total_reward_points": users["total_reward_points"],
                "following": users["following"],
                "followers": users["followers"],
                "referees": users["referees"],
                "referred_by": users["referred_by"],  # drop later
                "client_app_id": users["client_app_id"],  # drop later
                "user_id": users["_id"],  # drop later
            }
        )

        # remove any records where the status is deleted or not verified
        metrics = metrics[metrics["status"] != "Deleted"]
        metrics = metrics[metrics["status"] != "Not Verified"]

        self._processed_data["user_data"] = metrics

    def process_sessions(self):
        """Process session analysis."""
        rewards = self._raw_data["rewards"].copy()
        users = self._raw_data["users"].copy()

        # Create username lookup dictionary
        username_map = users.set_index("_id")["username"].to_dict()

        # Filter for session-relevant activity (points = 0)
        session_rewards = rewards[(rewards["points"] >= 0)].sort_values("created_atAdj")

        # map the team name to the client_app_id
        session_rewards["team_name"] = session_rewards["client_app_id"].map(
            self.team_name_mapping()
        )

        sessions_list = []

        # Group by user
        for user_id, user_rewards in session_rewards.groupby("sender_id"):
            # Get all session starts (OPEN_APP or WELCOME events)
            session_starts = user_rewards[
                user_rewards["type"].isin(["OPEN_APP", "WELCOME"])
            ].copy()

            if session_starts.empty:
                continue

            # Round timestamps to minutes and group by minute
            session_starts["minute"] = session_starts["created_atAdj"].dt.floor("min")

            # Get the minimum timestamp within each minute
            session_starts = (
                session_starts.groupby("minute")["created_atAdj"].min().tolist()
            )

            # Add artificial end to capture last session
            session_starts.append(pd.Timestamp.now(tz=self.offset))

            # Process each session
            for i in range(len(session_starts) - 1):
                session_start = session_starts[i]
                session_end = session_starts[i + 1]

                # get the time between the session_start and session_end
                max_time_between = session_end - session_start

                # Get rewards in this session
                session_rewards = user_rewards[
                    (user_rewards["created_atAdj"] >= session_start)
                    & (user_rewards["created_atAdj"] < session_end)
                ]

                # count the number of OPEN_APP and WELCOME events
                open_app_count = len(
                    session_rewards[session_rewards["type"] == "OPEN_APP"]
                )
                welcome_count = len(
                    session_rewards[session_rewards["type"] == "WELCOME"]
                )

                # Calculate session metrics
                practice_count = len(
                    session_rewards[session_rewards["type"] == "PRACTICE"]
                )
                raw_duration = (
                    session_rewards["created_atAdj"].max()
                    - session_rewards["created_atAdj"].min()
                ).total_seconds()

                # Adjust duration for practice rewards (120 seconds each)
                adjusted_duration_seconds = int(
                    raw_duration + (practice_count + 1 * 60)
                )

                # if the adjusted_duration_Seconds > max_time_between, then set adjusted_duration_seconds to max_time_between
                if adjusted_duration_seconds > max_time_between.total_seconds():
                    adjusted_duration_seconds = max_time_between.total_seconds()

                adjusted_duration_minutes = int(
                    round(adjusted_duration_seconds / 60, 0)
                )

                sessions_list.append(
                    {
                        "username": username_map.get(user_id, "Unknown"),
                        "team_name": session_rewards["team_name"].iloc[0],
                        "session_start": session_start,
                        "last_activity": session_rewards["created_atAdj"].max(),
                        "session_end": session_start
                        + pd.Timedelta(seconds=adjusted_duration_seconds),
                        "session_length_seconds": adjusted_duration_seconds,
                        "session_length_minutes": adjusted_duration_minutes,
                        "reward_count": len(session_rewards)
                        - open_app_count
                        - welcome_count,
                        "practice_count": practice_count,
                        "client_app_id": session_rewards["client_app_id"].iloc[0],
                        "user_id": user_id,
                    }
                )

        sessions_df = pd.DataFrame(sessions_list)
        sessions_df = sessions_df[sessions_df["username"] != "risingadmin"]
        self._processed_data["sessions"] = sessions_df

    def process_engagement(self):
        """Process user engagement metrics.

        Returns DataFrame with columns:
        - sender_id
        - username
        - team_name
        - client_app_id
        - time
        - timestamp
        - datestamp
        - engagement_type
        - points
        - receiver_id
        Includes engagement types:
        - SIGNUP
        - OPEN_APP
        - WELCOME
        - PRACTICE
        - CHEER
        - CREATOR
        - CONNECT
        - EXPRESS
        - CHAT
        - SHOP
        - REFER
        """
        rewards = self._raw_data["rewards"].copy()
        users = self._raw_data["users"].copy()

        if rewards.empty:
            print("No rewards data available for engagement analysis")
            self._processed_data["engagement"] = pd.DataFrame()
            return

        # Create username lookup
        username_map = users.set_index("_id")["username"].to_dict()

        # Define engagement types to include
        engagement_types = [
            "SIGNUP",
            "OPEN_APP",
            "WELCOME",
            "PRACTICE",
            "CHEER",
            "CREATOR",
            "CONNECT",
            "EXPRESS",
            "CHAT",
            "SHOP",
            "REFER",
            # "ENGAGE", These are recevied
            # "INFLUENCE", These are recevied
        ]

        # Filter for specified engagement types
        engagements = rewards[rewards["type"].isin(engagement_types)].copy()

        # Add username
        engagements["username"] = engagements["sender_id"].map(username_map)
        engagements["receiver_username"] = engagements["receiver_id"].map(username_map)
        engagements["team_name"] = engagements["client_app_id"].map(
            self.team_name_mapping()
        )

        # Select and rename columns
        engagement_df = engagements[
            [
                "username",
                "team_name",
                "client_app_id",
                "created_at",
                "created_atAdj",
                "created_atAdjDate",
                "type",
                "points",
                "sender_id",
                "receiver_id",
                "receiver_username",
            ]
        ].rename(
            columns={
                "username": "sender_username",
                # "created_at": "time",
                # "created_atAdj": "timestamp",
                "created_atAdjDate": "datestamp",
                "type": "engagement_type",
            }
        )

        # filter for points > 0
        rewards_df = engagement_df[engagement_df["points"] > 0]

        # filter for only engagements with 0 points
        # engagement_df = engagement_df[engagement_df["points"] == 0]

        # Store results
        self._processed_data["engagement"] = engagement_df
        self._processed_data["rewards"] = rewards_df

    def process_subscriptions(self):
        """Process subscription data.

        Returns DataFrame with columns:
        - user_id
        - sub_type
        - duration
        - is_active
        - platform
        - team_name
        """
        subscriptions = self._raw_data["subscriptions"].copy()
        users = self._raw_data["users"].copy()

        if subscriptions.empty:
            print("No subscription data available")
            self._processed_data["subscriptions"] = pd.DataFrame()
            return

        # Create username lookup
        username_map = users.set_index("_id")["username"].to_dict()

        # drop rows where payment_status is 'failed'
        subscriptions = subscriptions[subscriptions["payment_status"] != "failed"]

        # drop rows where the amount is 3400
        subscriptions = subscriptions[subscriptions["amount"] != 3400]

        # drop the payment_status column
        subscriptions = subscriptions.drop(columns=["payment_status"])

        # Select relevant columns
        sub_df = subscriptions[
            [
                "user_id",
                "sub_type",
                "duration",
                "is_active",
                "platform",
                "client_app_id",
                "amount",
            ]
        ].copy()

        # Clean up data
        sub_df["is_active"] = sub_df["is_active"].fillna(False)
        sub_df["username"] = sub_df["user_id"].map(username_map)
        sub_df["team_name"] = sub_df["client_app_id"].map(self.team_name_mapping())

        # drop the client_app_id
        sub_df = sub_df.drop(columns=["client_app_id"])

        # drop na usernames
        sub_df = sub_df[sub_df["username"].notna()]

        # drop risingadmin
        sub_df = sub_df[sub_df["username"] != "risingadmin"]

        # drop rows with username _amakiri
        sub_df = sub_df[sub_df["username"] != "amakiri__"]

        # drop rows with username kirinpatel
        sub_df = sub_df[sub_df["username"] != "kirinpatel"]

        # remove any rows where the amount > 100
        sub_df = sub_df[sub_df["amount"] <= 100]

        # Store results
        self._processed_data["subscriptions"] = sub_df

        # Add subscription data to user profiles if they exist
        if (
            "user_data" in self._processed_data
            and not self._processed_data["user_data"].empty
        ):
            users = self._processed_data["user_data"]
            users_with_subs = users.merge(
                sub_df,
                left_on="user_id",
                right_on="user_id",
                how="left",
            )

            # drop duplicates keeping last
            users_with_subs = users_with_subs.drop_duplicates(
                subset=["user_id"], keep="last"
            )

            # clean column names
            cols_to_drop = [col for col in users_with_subs.columns if "_y" in col]
            users_with_subs = users_with_subs.drop(columns=cols_to_drop)

            # rename the columns
            cols_to_rename = [col for col in users_with_subs.columns if "_x" in col]
            users_with_subs = users_with_subs.rename(
                columns={col: col.replace("_x", "") for col in cols_to_rename}
            )

            # Update user profiles
            self._processed_data["user_data"] = users_with_subs

    def process_retention(self, engagement_df, time_type="weekly", backstop_days=45):
        """Calculate user retention from engagement data."""
        # Create a copy to avoid modifying the original dataframe
        df = engagement_df.copy()

        # # Ensure timestamp is in datetime format with timezone
        df["created_atAdj"] = pd.to_datetime(df["created_atAdj"])

        # Get first engagement date for each user
        first_engagement = df.groupby("sender_id")["created_atAdj"].transform("min")

        # Create cohort columns
        df["CohortDay"] = first_engagement.dt.strftime("%Y-%m-%d")
        df["CohortDayIndex"] = (df["created_atAdj"] - first_engagement).dt.days + 1

        # Create week-based cohorts
        df["CohortWeek"] = first_engagement.dt.strftime("%Y-%U")
        df["ActivityWeek"] = df["created_atAdj"].dt.strftime("%Y-%U")

        # Calculate week index similar to your original method
        df["CohortWeekIndex"] = (
            (
                (
                    df["ActivityWeek"].str[:4].astype(int)
                    - df["CohortWeek"].str[:4].astype(int)
                )
                * 52
            )
            + (
                df["ActivityWeek"].str[5:].astype(int)
                - df["CohortWeek"].str[5:].astype(int)
            )
            + 1
        )

        # Create month-based cohorts
        df["CohortMonth"] = first_engagement.dt.strftime("%Y-%m")
        df["ActivityMonth"] = df["created_atAdj"].dt.strftime("%Y-%m")

        # Calculate month index similar to your original method
        df["CohortMonthIndex"] = (
            (
                df["ActivityMonth"].str[:4].astype(int)
                - df["CohortMonth"].str[:4].astype(int)
            )
            * 12
            + (
                df["ActivityMonth"].str[5:].astype(int)
                - df["CohortMonth"].str[5:].astype(int)
            )
            + 1
        )

        # Set counter and indexer based on time_type
        counter = (
            "CohortDay"
            if time_type == "daily"
            else "CohortWeek" if time_type == "weekly" else "CohortMonth"
        )
        indexer = (
            "CohortDayIndex"
            if time_type == "daily"
            else "CohortWeekIndex" if time_type == "weekly" else "CohortMonthIndex"
        )

        # Get current time and backstop
        today = pd.Timestamp.now(tz=self.offset)
        today = today.date()
        today = dt.datetime.combine(today, dt.datetime.min.time())
        backstop = today - dt.timedelta(days=backstop_days)

        # Create cohort analysis
        grouping = df.groupby([counter, indexer])
        cohorts = grouping["sender_id"].nunique().reset_index()
        cohort_counts = cohorts.pivot(
            index=counter, columns=indexer, values="sender_id"
        )

        if time_type == "daily":
            # Create complete date range
            date_range = pd.date_range(start=backstop, end=today, freq="D")
            date_range = date_range.strftime("%Y-%m-%d")

            # Create complete range for columns (1 to number of days)
            column_range = range(
                1, (today - backstop).days + 2
            )  # +2 to include both ends

            # Reindex both rows and columns
            cohort_counts = cohort_counts.reindex(
                index=date_range, columns=column_range
            )

        elif time_type == "weekly":
            # Create complete week range
            week_range = pd.date_range(start=backstop, end=today, freq="W")
            week_range = week_range.strftime("%Y-%U")

            # Create complete range for columns (1 to number of weeks)
            num_weeks = len(week_range)
            column_range = range(1, num_weeks + 1)

            # Reindex both rows and columns
            cohort_counts = cohort_counts.reindex(
                index=week_range, columns=column_range
            )

        elif time_type == "monthly":
            # Create complete month range
            month_range = pd.date_range(start=backstop, end=today, freq="M")
            month_range = month_range.strftime("%Y-%m")

            # Create complete range for columns (1 to number of months)
            num_months = len(month_range)
            column_range = range(1, num_months + 1)

            # Reindex both rows and columns
            cohort_counts = cohort_counts.reindex(
                index=month_range, columns=column_range
            )

        # Filter and fill missing values
        if time_type == "daily":
            cohort_counts = cohort_counts[
                pd.to_datetime(cohort_counts.index) >= backstop
            ]
        elif time_type == "weekly":
            cohort_counts = cohort_counts[
                cohort_counts.index >= backstop.strftime("%Y-%U")
            ]
        elif time_type == "monthly":
            backstop = backstop.strftime("%Y-%m")

            # filter the cohort_counts to only include the backstop days
            cohort_counts = cohort_counts[cohort_counts.index >= backstop]

            # iterate over each column and row of cohort_counts
            for i in range(cohort_counts.shape[0]):
                for j in range(cohort_counts.shape[1]):

                    # convert today to month
                    today_month = today.strftime("%Y-%m")

                    # today_month to an integer by removing the dash
                    today_month = int(today_month.replace("-", ""))

                    # if a value is na, and the month is less than the current month, change to 0
                    if int(cohort_counts.index[i].replace("-", "")) + j <= today_month:
                        # check if the value is na
                        if pd.isna(cohort_counts.iloc[i, j]):
                            cohort_counts.iloc[i, j] = 0

        # determine the retention rate
        cohort_sizes = cohort_counts.iloc[:, 0]

        # divide the cohort_counts by the cohort_sizes
        retention = cohort_counts.divide(cohort_sizes, axis=0)

        # round to 2 decimal places
        retention = retention.round(2)

        # multiply by 100 to get the percentage
        # retention = retention.round(3) * 100

        return cohort_counts, retention

    def process_rfmp(self):
        """Process RFM(P) analysis with subscription data."""
        sessions = self._processed_data["sessions"].copy()
        subscriptions = self._processed_data["subscriptions"].copy()
        users = self._processed_data["user_data"].copy()

        # filter out risingadmin
        sessions = sessions[sessions["username"] != "risingadmin"]
        subscriptions = subscriptions[subscriptions["username"] != "risingadmin"]
        users = users[users["username"] != "risingadmin"]

        if sessions.empty:
            print("No session data available for RFMP analysis")
            self._processed_data["rfmp"] = pd.DataFrame()
            return

        # Get current timestamp for recency calculation
        snapshot_date = pd.Timestamp.now(tz=self.offset)

        # Calculate base metrics
        rfmp = pd.DataFrame(index=sessions["user_id"].unique())

        # add date_joined to the rfmp dataframe
        rfmp["date_joined"] = users.set_index("user_id")["date_joined"]
        # convert date_joined to datetime
        rfmp["date_joined"] = pd.to_datetime(rfmp["date_joined"])

        # calculate recency as max session start date - snapshot date
        rfmp["recency"] = (
            sessions.groupby("user_id")["session_start"]
            .max()
            .map(lambda x: (snapshot_date - x).days)  # Changed column names
        )
        rfmp["frequency"] = sessions.groupby("user_id").size()
        rfmp["monetary_value"] = subscriptions.groupby("user_id")["amount"].sum()
        rfmp["total_points"] = users.groupby("user_id")["total_reward_points"].sum()
        # rfmp["avg_points"] = sessions.groupby("user_id")["points"].mean()

        def safe_score(series, reverse=False):
            """Safely score a series into quartiles using rank-based method.

            Args:
                series (pd.Series): Series to score
                reverse (bool): If True, higher values get lower scores

            Returns:
                pd.Series: Scored series with values 1-4
            """
            if series.empty or len(series.unique()) < 2:
                return pd.Series(1, index=series.index)

            # Rank the values (1 to n)
            ranks = series.rank(method="dense")

            # Scale ranks to 1-4 range
            min_rank = ranks.min()
            max_rank = ranks.max()

            if max_rank == min_rank:
                return pd.Series(1, index=series.index)

            scores = ((ranks - min_rank) / (max_rank - min_rank)) * 3 + 1

            # Round to nearest integer
            scores = scores.round()

            # Reverse scores if needed
            if reverse:
                scores = 5 - scores

            return scores

        # Score each metric
        rfmp["recency_score"] = safe_score(rfmp["recency"], reverse=True)
        rfmp["frequency_score"] = safe_score(rfmp["frequency"])
        rfmp["monetary_value_score"] = safe_score(rfmp["monetary_value"])
        rfmp["total_points_score"] = safe_score(rfmp["total_points"])

        # Calculate RFMP score
        rfmp["rfmp_score"] = (
            rfmp["recency_score"].fillna(1)
            + rfmp["frequency_score"].fillna(1)
            + rfmp["monetary_value_score"].fillna(1)
            + rfmp["total_points_score"].fillna(1)
        )

        # Assign RFMP level based on available range
        score_range = rfmp["rfmp_score"].max() - rfmp["rfmp_score"].min()
        if score_range < 1:
            rfmp["rfmp_level"] = "Medium"
        else:
            score_bins = [
                rfmp["rfmp_score"].min() - 0.1,  # Add small buffer for inclusive bins
                rfmp["rfmp_score"].min() + score_range / 3,
                rfmp["rfmp_score"].min() + 2 * score_range / 3,
                rfmp["rfmp_score"].max() + 0.1,  # Add small buffer for inclusive bins
            ]
            rfmp["rfmp_level"] = pd.cut(
                rfmp["rfmp_score"], bins=score_bins, labels=["Low", "Medium", "High"]
            )

        # Add user segments based on RFMP score
        try:
            rfmp["rfmp_segment"] = pd.qcut(
                rfmp["rfmp_score"],
                q=3,
                labels=["Bronze", "Silver", "Gold"],
                duplicates="drop",
            )
        except ValueError:
            # If we can't create three segments, assign all to Silver
            rfmp["rfmp_segment"] = "Silver"

        # Add user status
        rfmp["user_status"] = "Active"  # Default status
        rfmp.loc[rfmp["recency"] > 30, "user_status"] = "Latent"
        rfmp.loc[rfmp["recency"] > 90, "user_status"] = "Churned"

        # create a "status_date" column
        # if active, use date_joined
        # if latent, use recency - 30 from today
        # if churned, use recency - 90 from today
        rfmp["status_date"] = rfmp["date_joined"]
        rfmp.loc[rfmp["user_status"] == "Latent", "status_date"] = (
            snapshot_date - pd.to_timedelta(rfmp["recency"] - 30, unit="D")
        ).dt.tz_localize(None)
        rfmp.loc[rfmp["user_status"] == "Churned", "status_date"] = (
            snapshot_date - pd.to_timedelta(rfmp["recency"] - 90, unit="D")
        ).dt.tz_localize(None)
        # reduce to just date
        rfmp["status_date"] = rfmp["status_date"].dt.date

        # Update any remaining null user_status to "Inactive" and set the status_date to date_joined
        rfmp.loc[rfmp["user_status"].isna(), "user_status"] = "Inactive"

        # Create comprehensive user metrics dataframe
        user_metrics = users.copy()
        user_metrics = user_metrics.merge(
            rfmp[
                [
                    "recency",
                    "frequency",
                    "monetary_value",
                    "total_points",
                    # "avg_points",
                    "rfmp_score",
                    "rfmp_level",
                    "rfmp_segment",
                    "user_status",
                    "status_date",
                ]
            ],
            left_on="user_id",
            right_index=True,
            how="left",
        )

        # Update "user_status" to "Not Verified" if status is "Not Verified
        user_metrics.loc[user_metrics["status"] == "Not Verified", "user_status"] = (
            "Not Verified"
        )

        # fill null user_status with "Inactive"
        user_metrics.loc[user_metrics["user_status"].isna(), "user_status"] = "Inactive"

        # fill null status_date with date_joined
        user_metrics.loc[user_metrics["status_date"].isna(), "status_date"] = (
            user_metrics["date_joined"]
        )

        # Store results
        self._processed_data["rfmp"] = rfmp
        self._processed_data["user_data"] = user_metrics

    def get_topline_metrics(self, users, engagement, subscriptions):
        """Get topline metrics for the team."""
        # Get data
        users_df = users
        engagement_df = engagement
        subscriptions_df = subscriptions

        # Get today and yesterday dates in correct timezone
        today = pd.Timestamp.now(tz=self.offset).date()
        yesterday = today - pd.Timedelta(days=1)

        # filter out "Not Verified" from the users_df
        users_df = users_df[users_df["user_status"] != "Not Verified"]
        users_df = users_df[users_df["user_status"] != "Inactive"]

        # User metrics
        total_users = len(users_df) if not users_df.empty else 0

        # New users (using date_joined from users)
        if not users_df.empty:
            # Convert date_joined to datetime if it's not already
            users_df["date_joined"] = pd.to_datetime(users_df["date_joined"])
            new_users_today = len(users_df[users_df["date_joined"].dt.date == today])
            new_users_yesterday = len(
                users_df[users_df["date_joined"].dt.date == yesterday]
            )
        else:
            new_users_today = 0
            new_users_yesterday = 0

        # Engagement metrics
        if not engagement_df.empty:
            total_engagements = len(engagement_df)
            eng_today = len(engagement_df[engagement_df["datestamp"] == today])
            eng_yesterday = len(engagement_df[engagement_df["datestamp"] == yesterday])

            # Active users
            active_users_today = engagement_df[engagement_df["datestamp"] == today][
                "sender_id"
            ].nunique()

            active_users_yesterday = engagement_df[
                engagement_df["datestamp"] == yesterday
            ]["sender_id"].nunique()
        else:
            total_engagements = 0
            eng_today = 0
            eng_yesterday = 0
            active_users_today = 0
            active_users_yesterday = 0

        # Subscription metrics
        if not subscriptions_df.empty:
            active_subs = len(users_df[users_df["is_active"] == True])
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

    def process_network(self):
        """Process network analysis."""
        users = self._raw_data["users"].copy()
        rewards = self._raw_data["rewards"].copy()

        network_data = []

        # Process followership
        for _, user in users.iterrows():
            # Add follower relationships
            for follower in user.get("followers", []):
                network_data.append(
                    {
                        "sender_id": user["_id"],
                        "receiver_id": follower,
                        "engagement_type": "follow",
                        "engagement_value": 0,
                        "edges": 1,
                    }
                )

            # Add referral relationships
            if user.get("referred_by"):
                network_data.append(
                    {
                        "sender_id": user["referred_by"],
                        "receiver_id": user["_id"],
                        "engagement_type": "referral",
                        "engagement_value": 0,
                        "edges": 1,
                    }
                )

        # Process engagement rewards
        network_types = ["CHEER", "EXPRESS"]

        # filter to the last 60 days
        rewards = rewards[
            rewards["created_at"]
            >= (pd.Timestamp.now(tz=self.offset) - pd.Timedelta(days=60))
        ]

        network_rewards = rewards[rewards["type"].isin(network_types)]

        for _, reward in network_rewards.iterrows():
            network_data.append(
                {
                    "sender_id": reward["sender_id"],
                    "receiver_id": reward["receiver_id"],
                    "engagement_type": reward["type"].lower(),
                    "engagement_value": reward["points"],
                    "edges": 1,
                }
            )

        # Create network DataFrame
        network_df = pd.DataFrame(network_data)

        # Add usernames
        username_map = users.set_index("_id")["username"].to_dict()
        network_df["sender_username"] = network_df["sender_id"].map(username_map)
        network_df["receiver_username"] = network_df["receiver_id"].map(username_map)

        # filter out risingadmin
        network_df = network_df[network_df["sender_username"] != "risingadmin"]
        network_df = network_df[network_df["receiver_username"] != "risingadmin"]

        self._processed_data["network"] = network_df

    def create_network_graph(self):
        """Create network graph of user interactions.

        Returns:
            G: networkx DiGraph
        """
        try:

            # Get network data
            network_df = self._processed_data.get("network")
            if network_df is None or network_df.empty:
                print("No network data available")
                return None

            # Create directed graph
            G = nx.DiGraph()

            # Add edges with weights
            for _, row in network_df.iterrows():
                source = row["sender_username"]
                target = row["receiver_username"]

                # Skip if either username is None
                if pd.isna(source) or pd.isna(target):
                    continue

                # Add nodes if they don't exist
                if not G.has_node(source):
                    G.add_node(source, type="user")
                if not G.has_node(target):
                    G.add_node(target, type="user")

                # Add or update edge
                if G.has_edge(source, target):
                    G[source][target]["weight"] += 1
                else:
                    G.add_edge(source, target, weight=1)

            # convert graph to non-directed
            G = G.to_undirected()

            if len(G.nodes()) == 0:
                print("No valid connections found in the network")
                return None

            return G

        except Exception as e:
            print(f"Error creating network graph: {e}")
            return None

    def assign_network_properties(self, g, color=True, node_cutoff=30, node_remove=60):
        """Assign properties to nodes and edges of networkx graph

        Args:
            g: networkx graph
            color: boolean, whether to color nodes by community
            node_cutoff: int, number of days to cutoff for node color
            node_remove: int, number of days to remove a node from the network

        Returns:
            g: networkx graph with properties assigned
            communities: list of communities
        """
        # First, remove inactive nodes
        nodes_to_remove = []
        for node_id in g.nodes:
            try:
                recency = (
                    self._processed_data["user_data"]
                    .set_index("username")["recency"]
                    .loc[node_id]
                )
                if recency > node_remove:
                    nodes_to_remove.append(node_id)
            except KeyError:
                nodes_to_remove.append(node_id)

        # Remove inactive nodes
        for node_id in nodes_to_remove:
            g.remove_node(node_id)

        # Now calculate network metrics with cleaned graph
        node_degrees = nx.degree_centrality(g)
        edge_centralities = nx.edge_betweenness_centrality(g)
        node_page_rank = nx.pagerank(g)
        node_centralities = nx.centrality.betweenness_centrality(g)
        node_degree = nx.degree(g)

        # Community detection
        communities = nx.algorithms.community.greedy_modularity_communities(g)

        # Graph properties
        g.graph["node_border_size"] = 1.5
        g.graph["node_border_color"] = "white"
        g.graph["edge_opacity"] = 0.9

        # Node properties: Size by centrality, shape by size, color by community
        colors = [
            "red",
            "blue",
            "green",
            "orange",
            "pink",
            "brown",
            "yellow",
            "cyan",
            "magenta",
            "violet",
        ]

        for node_id in g.nodes:
            try:
                recency = (
                    self._processed_data["user_data"]
                    .set_index("username")["recency"]
                    .loc[node_id]
                )

                node = g.nodes[node_id]
                node["user_status"] = recency
                node["size"] = 10 + node_page_rank[node_id] * 100
                node["centrality"] = node_centralities[node_id]
                node["degrees"] = node_degrees[node_id]
                node["page_rank"] = node_page_rank[node_id]
                node["degree"] = node_degree[node_id]
                node["shape"] = "circle"
                community_counter = 0  # Initialize default value
                for community_counter, community_members in enumerate(communities):
                    if node_id in community_members:
                        break
                if recency < node_cutoff:
                    node["color"] = colors[community_counter % len(colors)]
                else:
                    node["color"] = "lightgray"
                    node["label_color"] = "lightgray"
            except KeyError:
                continue

        # Edge properties: Size by centrality, color by community
        for edge_id in g.edges:
            edge = g.edges[edge_id]
            source_node = g.nodes[edge_id[0]]
            target_node = g.nodes[edge_id[1]]
            edge["size"] = edge_centralities[edge_id] * 100
            edge["color"] = (
                source_node["color"]
                if source_node["color"] == target_node["color"]
                else "black"
            )

        if color == False:
            node["color"] = "black"

        return g, communities

    def process_team_data(self):
        """Process user data for team download."""
        # Get team data
        team_data = self._processed_data["user_data"].copy()

        # select the columns to keep
        team_data = team_data[
            [
                "username",
                "status",
                "team_name",
                "email",
                "phone_number",
                "dob",
                "country",
                "date_joined",
                "total_reward_points",
                "sub_type",
                "duration",
                "is_active",
                "platform",
            ]
        ]

        # Store results
        self._processed_data["team_data"] = team_data

    def run_pipeline(self):
        """Run the complete analytics pipeline"""

        # Process all analyses
        self.team_name_mapping()
        self.process_client_data()
        self.process_user_data()
        self.process_sessions()
        self.process_engagement()
        self.process_subscriptions()
        self.process_network()
        self.process_rfmp()
        self.process_team_data()

        return self._processed_data
