# ParlayBud

**Advanced NBA Parlay Analysis Tool with Positive Expected Value Detection**

ParlayBud is a sports betting analytics platform that helps identify profitable NBA parlay opportunities using statistical modeling, Monte Carlo simulations, and correlation analysis.

![ParlayBud](https://img.shields.io/badge/React-18.3-blue?logo=react)
![TypeScript](https://img.shields.io/badge/TypeScript-5.0-blue?logo=typescript)
![Tailwind CSS](https://img.shields.io/badge/Tailwind-3.4-38B2AC?logo=tailwindcss)

![Check it out here](https://parlaybud.lovable.app)

## Features

- ** Advanced Statistical Engine**
  - Monte Carlo copula-based simulations for correlated outcomes
  - Hierarchical Bayesian player minutes estimation
  - Position-based correlation modeling
  - Probability blending from historical data and projections

- ** +EV Parlay Detection**
  - Identifies parlays with positive expected value
  - Kelly Criterion-based bet sizing recommendations
  - Model vs. market probability edge analysis

- ** Modern UI/UX**
  - Multiple view modes: Detailed, Compact, and List
  - Dark/Light theme toggle
  - Frosted glass design aesthetic
  - Responsive mobile-friendly layout

- ** Data Management**
  - Upload Excel files with player props data
  - Import/export JSON datasets
  - Multi-select and bulk copy parlays
  - Dataset renaming and deletion

## How It Works

1. **Upload** an Excel file containing NBA player props data
2. **Process** through our parlay engine which:
   - Preprocesses and validates props data
   - Estimates player minutes using hierarchical modeling
   - Computes model probabilities from historical hit rates & projections
   - Builds a correlation matrix between player propositions
   - Runs Monte Carlo simulations (10,000+ trials)
   - Searches for parlays with positive expected value
3. **Analyze** the generated parlays with detailed leg breakdowns
4. **Copy** your selections and place informed bets

## Tech Stack

- **Frontend:** React 18, TypeScript, Vite
- **Styling:** Tailwind CSS, shadcn/ui
- **Backend:** Supabase Edge Functions
- **Database:** PostgreSQL (via Supabase)
- **Data Processing:** XLSX parsing

## Statistics Displayed

| Metric | Description |
|--------|-------------|
| Total Parlays | Number of +EV parlays generated |
| Avg EV | Average expected value percentage |
| Avg Hit Prob | Average probability of parlay hitting |

## Excel File Format

Your input Excel file should contain columns like:
- `Player`, `Team`, `Game`, `Position`
- `Proposition`, `Line`, `Odds`
- `Last 5/10/20 Hit Rate`, `2025 Hit Rate`
- `Projection`, `StdDev`, `LastMinutes`

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
