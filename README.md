# Quantitative Trading Competition for Course Final Project


## 1. Competition Timeline



## 2. Dataset & Universe

### Asset Universe
*   **Selection:** All US equities from S&P 500 with sufficient historical data will be selected.
*   **Survivorship Bias:** Delisted stocks are excluded. While this introduces survivorship bias, it is acceptable for the scope of this academic project.
*   **Corporate Actions:** All price data will be fully adjusted for stock splits and dividends.

### Data Frequency & Splits
*   **Frequency:** 5-minute OHLCV data (subject to change to 1-minute or 1-hour).
*   **Training Data (Public):** Approximately the most recent 6 years of data.
*   **Public Test Data (Hidden on Kaggle):** Approximately 6 months of data immediately following the training set.
*   **Private Test Data (Live):** 3 weeks of live data generated during the exam period.

---

## 3. Trading Rules & Constraints

Students will submit a portfolio allocation function. At each timestamp $t$, the function receives historical data and outputs a vector of target portfolio weights.

Let $N$ be the total number of tradable assets. Let $w_{i,t}$ represent the target weight of asset $i$ at time $t$.

### Constraints
1.  **No Short Selling:** Weights must be non-negative.
    $$0 \le w_{i,t} \le 1 \quad \text{for all } i \in \{1, \dots, N\}$$
2.  **Maximum Exposure:** The sum of all asset weights cannot exceed 1 (no leverage).
    $$\sum_{i=1}^{N} w_{i,t} \le 1$$
3.  **Cash Allocation:** Any unallocated weight is implicitly held as cash. The weight of cash is defined as:
    $$w_{\text{cash},t} = 1 - \sum_{i=1}^{N} w_{i,t}$$
    The return on cash is set to $0\%$.
4.  **Market Frictions:** To simplify the environment for students, **transaction costs and slippage are assumed to be zero**. 

---

## 4. Submission & Execution Mechanism

*   **Platform:** Kaggle (Code Competition format).
*   **Submission Format:** Students submit a Kaggle Notebook containing their strategy code. No local submissions are permitted.
*   **Execution State & Data Gap:** 
    *   During the 3-week live evaluation phase, the Kaggle environment will **only** feed the new live data to the students' notebooks.
    *   The historical Training and Public Test datasets will *not* be available at runtime during the live phase.
    *   *Implication:* Students must pre-train their models (e.g., machine learning weights, statistical parameters) locally or in their training notebooks, and hardcode/load these learned parameters into their final submission script. The script must be able to generate allocations using only the live data points provided.

---

## 5. Evaluation Metric

For simplicity and clarity, the sole quantitative metric for the competition will be the **Sharpe Ratio**. 

Let $R_{i,t}$ be the return of asset $i$ at time $t$. The total portfolio return $R_{p,t}$ at time $t$ is calculated as the sum of the returns of the individual assets weighted by the allocations chosen at the *previous* timestamp $t-1$:

$$R_{p,t} = \sum_{i=1}^{N} w_{i,t-1} R_{i,t}$$

*(Note: Because cash earns 0%, the cash weight does not contribute to the portfolio return).*

Let $\mu_p$ be the sample mean of the portfolio returns over the evaluation period, and $\sigma_p$ be the sample standard deviation of the portfolio returns:

$$\mu_p = \frac{1}{T} \sum_{t=1}^{T} R_{p,t}$$
$$\sigma_p = \sqrt{ \frac{1}{T-1} \sum_{t=1}^{T} (R_{p,t} - \mu_p)^2 }$$

The annualized **Sharpe Ratio (SR)** is defined as:

$$\text{SR} = \frac{\mu_p}{\sigma_p} \times \sqrt{F}$$

Where $F$ is the annualization factor based on the data frequency. For example, assuming 252 trading days per year and 78 5-minute bars per standard trading day (9:30 AM - 4:00 PM), $F = 252 \times 78 = 19,656$.

Strategies will be strictly ranked by this Sharpe Ratio on the Private Leaderboard at the end of the 3-week live period.