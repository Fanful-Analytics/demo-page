# Fanful MyTeam Analytics - Demo App

This is a standalone demo application for showcasing Fanful MyTeam Analytics.

## Directory Structure

```
demo_page/
├── demo.py              # Main demo application
├── src/                 # Source code modules
│   ├── __init__.py
│   ├── logic_data.py    # Data processing and analytics
│   ├── logic_visuals.py # Visualization functions
│   └── shared_components.py # Shared UI components
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Secrets

Create a `.streamlit/secrets.toml` file in the demo_page directory with the following structure:

```toml
[database]
server = "your_mongodb_connection_string"
offset = "US/Eastern"  # Your timezone

[demo]
password = "your_demo_password"
```

### 3. Run the App

```bash
streamlit run demo.py
```

## Features

- **Password Protection**: Secure access with password authentication
- **Network Graph**: Interactive visualization of user network communities
- **30-Day Retention Curve**: Daily retention analysis
- **Monthly Retention Curve**: Monthly retention analysis

## Deployment to Streamlit Cloud

1. Copy the entire `demo_page` directory to a new repository
2. In Streamlit Cloud, add your secrets in the app settings
3. Deploy from your repository

## Notes

- All necessary source files are included in the `src/` directory
- The app is completely self-contained and doesn't depend on parent directories
- Don't forget to copy your `.streamlit/config.toml` file if you have custom configurations
