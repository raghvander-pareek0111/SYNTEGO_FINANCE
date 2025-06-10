import streamlit as st
import pandas as pd
import Cohere
from datetime import datetime
import json
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Cohere API client
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
co = cohere.Client(COHERE_API_KEY)

# Load or initialize expenses and income DataFrame
def load_data(file_path="finance_data.csv"):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        return pd.DataFrame(columns=["Date", "Type", "Category", "Amount", "Description"])

# Save data to CSV
def save_data(df, file_path="finance_data.csv"):
    df.to_csv(file_path, index=False)

# Smart transaction formatting for AI analysis
def format_transaction(data):
    return f"{data['Type']}: {data['Amount']} on {data['Category']} - {data['Description']} (Date: {data['Date']})"

# AI-driven financial insights using Cohere
def get_financial_insight(query, transactions):
    if transactions.empty:
        return "No transactions available to analyze. Please add some transactions first."
    
    # Format transactions for the prompt
    formatted = "\n".join([format_transaction(row) for _, row in transactions.iterrows()])
    
    # Calculate totals for more accurate context
    total_income = transactions[transactions["Type"] == "Income"]["Amount"].sum()
    total_expense = transactions[transactions["Type"] == "Expense"]["Amount"].sum()
    expense_by_category = transactions[transactions["Type"] == "Expense"].groupby("Category")["Amount"].sum().to_dict()
    
    # Build the prompt with actual data
    prompt = (
        f"Financial Data:\n{formatted}\n\n"
        f"Summary:\nTotal Income: ${total_income:.2f}\nTotal Expenses: ${total_expense:.2f}\n"
        f"Expenses by Category: {expense_by_category}\n\n"
        f"User Query: {query}\nProvide accurate financial insights or advice based on the data:"
    )
    
    response = co.generate(
        model='command',
        prompt=prompt,
        max_tokens=200,
        temperature=0.7
    )
    return response.generations[0].text.strip()

# Send real-time notification
def send_notification(message):
    print(f"NOTIFICATION: {message}")
    webhook_url = "https://api.pushover.net/1/messages.json"
    try:
        requests.post(webhook_url, json={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": message
        })
    except Exception as e:
        print(f"Notification failed: {e}")

# Saving tips and budget alerts with consistent font
def check_saving_tips(df):
    if df.empty:
        return ["<div style='font-family: sans-serif; font-size: 14px; font-weight: normal;'>No transactions yet. Add some to get tips!</div>"]
    
    total_expense = df[df["Type"] == "Expense"]["Amount"].sum()
    total_income = df[df["Type"] == "Income"]["Amount"].sum()
    tips = []
    
    # Alert for 100% total spending
    if total_income > 0:
        spending_percentage = (total_expense / total_income) * 100
        if spending_percentage >= 100:
            tip = (
                "<div style='font-family: sans-serif; font-size: 14px; font-weight: normal;'>"
                "🚨 Critical Alert: You've spent <span style='font-family: sans-serif; font-size: 14px; font-weight: normal;'>100%</span> or more of your income! "
                "Total Income: <span style='font-family: sans-serif; font-size: 14px; font-weight: normal;'>${:.2f}</span>, "
                "Total Expenses: <span style='font-family: sans-serif; font-size: 14px; font-weight: normal;'>${:.2f}</span>. "
                "Prioritize cutting non-essential costs immediately."
                "</div>"
            ).format(total_income, total_expense)
            tips.append(tip)
            send_notification("Critical Alert: You've spent 100% or more of your income!")
        elif spending_percentage >= 80:
            tip = (
                "<div style='font-family: sans-serif; font-size: 14px; font-weight: normal;'>"
                "⚠️ Warning: You're spending over <span style='font-family: sans-serif; font-size: 14px; font-weight: normal;'>80%</span> of your income! "
                "Total Income: <span style='font-family: sans-serif; font-size: 14px; font-weight: normal;'>${:.2f}</span>, "
                "Total Expenses: <span style='font-family: sans-serif; font-size: 14px; font-weight: normal;'>${:.2f}</span>. "
                "Reduce discretionary spending to stay within budget."
                "</div>"
            ).format(total_income, total_expense)
            tips.append(tip)
            send_notification("Budget Alert: Spending exceeds 80% of income!")
    
    # Thresholds for category spending (50%, 60%, 70%, 80%, 90%, 100%)
    if total_income > 0:
        expense_by_category = df[df["Type"] == "Expense"].groupby("Category")["Amount"].sum()
        thresholds = [50, 60, 70, 80, 90, 100]
        
        for category, amount in expense_by_category.items():
            for threshold in thresholds:
                threshold_amount = total_income * (threshold / 100)
                if amount >= threshold_amount:
                    leftover_percentage = 100 - (amount / total_income * 100)
                    leftover_amount = total_income * (leftover_percentage / 100)
                    
                    # Base allocations (adjust based on severity of overspending)
                    essentials_percent = 60 if threshold <= 70 else 70
                    savings_percent = 20 if threshold <= 70 else 15
                    discretionary_percent = 20 if threshold <= 70 else 15
                    
                    tip = (
                        "<div style='font-family: sans-serif; font-size: 14px; font-weight: normal;'>"
                        f"⚠️ You're spending <span style='font-family: sans-serif; font-size: 14px; font-weight: normal;'>${amount:.2f}</span> on {category}, "
                        f"which is more than <span style='font-family: sans-serif; font-size: 14px; font-weight: normal;'>{threshold}%</span> of your income "
                        f"(<span style='font-family: sans-serif; font-size: 14px; font-weight: normal;'>${total_income:.2f}</span>). "
                        f"You have <span style='font-family: sans-serif; font-size: 14px; font-weight: normal;'>{leftover_percentage:.1f}%</span> of your income left "
                        f"(<span style='font-family: sans-serif; font-size: 14px; font-weight: normal;'>${leftover_amount:.2f}</span>). Here's how to manage it:<br>"
                        f"- <b>Save and Spend Efficiently:</b><br>"
                        f"  - Essentials (e.g., Bills, Food): Allocate ~<span style='font-family: sans-serif; font-size: 14px; font-weight: normal;'>{essentials_percent}%</span> "
                        f"(<span style='font-family: sans-serif; font-size: 14px; font-weight: normal;'>${leftover_amount * (essentials_percent / 100):.2f}</span>).<br>"
                        f"  - Savings or Debt Repayment: Allocate ~<span style='font-family: sans-serif; font-size: 14px; font-weight: normal;'>{savings_percent}%</span> "
                        f"(<span style='font-family: sans-serif; font-size: 14px; font-weight: normal;'>${leftover_amount * (savings_percent / 100):.2f}</span>).<br>"
                        f"  - Discretionary (e.g., Entertainment): Allocate ~<span style='font-family: sans-serif; font-size: 14px; font-weight: normal;'>{discretionary_percent}%</span> "
                        f"(<span style='font-family: sans-serif; font-size: 14px; font-weight: normal;'>${leftover_amount * (discretionary_percent / 100):.2f}</span>).<br>"
                        f"- <b>Efficient Spending Tips:</b><br>"
                        f"  - Prioritize essential expenses over discretionary ones.<br>"
                        f"  - Reduce spending in {category} by finding cheaper alternatives or cutting unnecessary costs.<br>"
                        f"  - Set a monthly budget for {category} to stay below <span style='font-family: sans-serif; font-size: 14px; font-weight: normal;'>{threshold}%</span> of your income."
                        "</div>"
                    )
                    tips.append(tip)
                    send_notification(f"Alert: Spending on {category} exceeds {threshold}% of income!")
                    break  # Only show the highest threshold exceeded
    
    tips.append("<div style='font-family: sans-serif; font-size: 14px; font-weight: normal;'>💡 Tip: Save 20% of your income for emergencies and investments.</div>")
    return tips

# Streamlit UI with Professional Design
st.markdown(
    "<h1 style='text-align: center; font-family: sans-serif; color: #2C3E50;'>Syntego Finance: AI-Powered Expense Tracker 🚀</h1>",
    unsafe_allow_html=True
)

# Initialize data
if "data" not in st.session_state:
    st.session_state.data = load_data()

# Sidebar for adding transactions
with st.sidebar:
    st.markdown(
        "<h3 style='font-family: sans-serif; color: #2C3E50;'>➕ Add Transaction</h3>",
        unsafe_allow_html=True
    )
    with st.expander("New Transaction", expanded=True):
        trans_type = st.selectbox("Type", ["Expense", "Income"], help="Choose transaction type")
        category = st.selectbox("Category", ["Food", "Transport", "Bills", "Salary", "Other"], help="Select a category")
        amount = st.number_input("Amount ($)", min_value=0.0, step=0.01, help="Enter the amount")
        description = st.text_input("Description", help="Add a brief note")
        if st.button("Add Transaction", key="add_btn", help="Save the transaction"):
            if amount > 0 and description:
                new_entry = pd.DataFrame({
                    "Date": [datetime.now().strftime("%Y-%m-d %H:%M:%S")],
                    "Type": [trans_type],
                    "Category": [category],
                    "Amount": [amount],
                    "Description": [description]
                })
                st.session_state.data = pd.concat([st.session_state.data, new_entry], ignore_index=True)
                save_data(st.session_state.data)
                send_notification(f"New {trans_type} added: ${amount} for {description}")
                st.success("Transaction added successfully! 🎉")
            else:
                st.error("Please enter a valid amount and description.")

# Transactions Section
st.markdown(
    "<h3 style='font-family: sans-serif; color: #2C3E50;'>📋 Your Transactions</h3>",
    unsafe_allow_html=True
)
st.dataframe(st.session_state.data, use_container_width=True)

# Delete Transactions
with st.expander("🗑️ Delete Transactions"):
    selected = st.multiselect("Select transactions to delete", st.session_state.data.index, help="Choose rows to remove")
    if st.button("Delete Selected", key="delete_btn"):
        if selected:
            st.session_state.data = st.session_state.data.drop(selected)
            save_data(st.session_state.data)
            send_notification("Transaction(s) deleted")
            st.success("Selected transactions deleted! ✅")
        else:
            st.warning("No transactions selected.")

# AI Chatbot (SynBot)
st.markdown(
    "<h3 style='font-family: sans-serif; color: #2C3E50;'>🤖 SynBot: AI Financial Assistant</h3>",
    unsafe_allow_html=True
)
with st.expander("Ask SynBot", expanded=True):
    query = st.text_input("Ask about your finances", placeholder="E.g., 'How much did I spend on food?'")
    if st.button("Ask SynBot", key="synbot_btn"):
        if query:
            insight = get_financial_insight(query, st.session_state.data)
            st.markdown(
                "<b style='font-family: sans-serif; font-size: 14px;'>SynBot Response:</b>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<div style='font-family: sans-serif; font-size: 14px; font-weight: normal;'>{insight}</div>",
                unsafe_allow_html=True
            )
        else:
            st.error("Please enter a question for SynBot.")

# Financial Overview
st.markdown(
    "<h3 style='font-family: sans-serif; color: #2C3E50;'>💰 Financial Overview</h3>",
    unsafe_allow_html=True
)
if not st.session_state.data.empty:
    total_income = st.session_state.data[st.session_state.data["Type"] == "Income"]["Amount"].sum()
    total_expense = st.session_state.data[st.session_state.data["Type"] == "Expense"]["Amount"].sum()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            "<b style='font-family: sans-serif; font-size: 14px;'>Total Income</b>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<h3 style='font-family: sans-serif; color: #4ECDC4;'>${total_income:.2f}</h3>",
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            "<b style='font-family: sans-serif; font-size: 14px;'>Total Expense</b>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<h3 style='font-family: sans-serif; color: #FF6B6B;'>${total_expense:.2f}</h3>",
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            "<b style='font-family: sans-serif; font-size: 14px;'>Net Balance</b>",
            unsafe_allow_html=True
        )
        balance = total_income - total_expense
        color = "#4ECDC4" if balance >= 0 else "#FF6B6B"
        st.markdown(
            f"<h3 style='font-family: sans-serif; color: {color};'>${balance:.2f}</h3>",
            unsafe_allow_html=True
        )
else:
    st.info("No transactions yet. Add some to see your overview!")

# Saving Tips and Alerts
st.markdown(
    "<h3 style='font-family: sans-serif; color: #2C3E50;'>💡 Saving Tips & Alerts</h3>",
    unsafe_allow_html=True
)
tips = check_saving_tips(st.session_state.data)
for tip in tips:
    st.markdown(tip, unsafe_allow_html=True)

# Chart for Expenses and Income
st.markdown(
    "<h3 style='font-family: sans-serif; color: #2C3E50;'>📊 Expense and Income by Category</h3>",
    unsafe_allow_html=True
)
if not st.session_state.data.empty:
    chart_data = st.session_state.data.groupby(["Type", "Category"])["Amount"].sum().reset_index()
    expenses = chart_data[chart_data["Type"] == "Expense"]
    income = chart_data[chart_data["Type"] == "Income"]
    
    st.markdown(
        "<h4 style='font-family: sans-serif; color: #2C3E50;'>Bar Chart</h4>",
        unsafe_allow_html=True
    )
    
    # Prepare data for st.bar_chart
    categories = ["Food", "Transport", "Bills", "Salary", "Other"]
    expense_data = [
        expenses[expenses['Category'] == 'Food']['Amount'].sum() if not expenses[expenses['Category'] == 'Food'].empty else 0,
        expenses[expenses['Category'] == 'Transport']['Amount'].sum() if not expenses[expenses['Category'] == 'Transport'].empty else 0,
        expenses[expenses['Category'] == 'Bills']['Amount'].sum() if not expenses[expenses['Category'] == 'Bills'].empty else 0,
        expenses[expenses['Category'] == 'Salary']['Amount'].sum() if not expenses[expenses['Category'] == 'Salary'].empty else 0,
        expenses[expenses['Category'] == 'Other']['Amount'].sum() if not expenses[expenses['Category'] == 'Other'].empty else 0
    ]
    income_data = [
        income[income['Category'] == 'Food']['Amount'].sum() if not income[income['Category'] == 'Food'].empty else 0,
        income[income['Category'] == 'Transport']['Amount'].sum() if not income[income['Category'] == 'Transport'].empty else 0,
        income[income['Category'] == 'Bills']['Amount'].sum() if not income[income['Category'] == 'Bills'].empty else 0,
        income[income['Category'] == 'Salary']['Amount'].sum() if not income[income['Category'] == 'Salary'].empty else 0,
        income[income['Category'] == 'Other']['Amount'].sum() if not income[income['Category'] == 'Other'].empty else 0
    ]

    # Create a DataFrame for st.bar_chart
    chart_df = pd.DataFrame({
        "Category": categories,
        "Expenses": expense_data,
        "Income": income_data
    })
    chart_df.set_index("Category", inplace=True)

    # Render the chart using st.bar_chart
    st.bar_chart(chart_df, use_container_width=True)
else:
    st.info("Add transactions to see the chart!")
