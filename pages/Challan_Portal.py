import streamlit as st
import pandas as pd
import os
import base64
import boto3
import sqlite3

def set_bg_from_local(image_file):
    with open(image_file, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_string}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg_from_local("background.jpg")

st.markdown("""
    <style>
    .black-bg {
        background-color: black;
        color: orange;
        padding: 10px;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .title {
        background-color: black;
        color: white;
        font-size: 35px;
        text-align: center;
        padding: 10px;
        opacity:0.9;
        border-radius:20px;
    }
    </style>
    <div class="title">
    🚦 Challan Authority Portal
    </div>
    """, unsafe_allow_html=True)

CHALLAN_DB_PATH = "violations.db"
VEHICLE_OWNERS_DB_PATH = "vehicle_owner.csv"

FINE_MAP = {
    "Helmet Violation": 500,
    "Overspeeding": 1000,
    "Red Light Violation": 1200,
    "License Plate Missing": 800
}

sns = boto3.client('sns','ap-northeast-3')
SENDER_ID = 'CHALLAN'

try:
    conn = sqlite3.connect(CHALLAN_DB_PATH)
    df = pd.read_sql_query("SELECT * FROM violations", conn)

    df = df[df["plate"].notna() & (df["plate"].str.strip().str.upper() != "NOT DETECTED") & (df["plate"].str.strip() != "")]

    df.rename(columns={
        "plate": "Number Plate",
        "violation_type": "Violation",
        "timestamp": "Date",
        "status": "Status"
    }, inplace=True)

    df["Challan ID"] = df["id"].apply(lambda x: f"CH-{int(x):05d}")

    df["Fine Amount"] = df["Violation"].map(FINE_MAP).fillna(500)

    owners_df = pd.read_csv(VEHICLE_OWNERS_DB_PATH)

    merged_df = pd.merge(df, owners_df, on="Number Plate", how="left")

    status_filter = st.radio("Filter by Payment Status:", ["All", "Paid", "Unpaid"])
    if status_filter != "All":
        merged_df = merged_df[merged_df["Status"].str.lower() == status_filter.lower()]

    for violation, group in merged_df.groupby("Violation"):
        st.markdown(f"""
<div style="margin:7px;color:white;" class="black-bg">
    <h4>{violation} Challans</h4>
</div>
""", unsafe_allow_html=True)
        for _, row in group.iterrows():
            st.markdown(f"""
<div class="black-bg">
    <strong>Challan ID:</strong> {row['Challan ID']}<br>
    <strong>Plate:</strong> {row['Number Plate']}<br>
    <strong>Fine:</strong> ₹{row['Fine Amount']}<br>
    <strong>Status:</strong> {row['Status']}<br>
    <strong>Date:</strong> {row['Date']}
</div>
""", unsafe_allow_html=True)
            st.markdown("--------")

    if st.button("Send SMS to Unpaid Challans Owners"):
        unpaid = merged_df[merged_df["Status"].str.lower() == "unpaid"]
        if unpaid.empty:
            st.warning("No unpaid challans.")
        else:
            for _, row in unpaid.iterrows():
                phone = str(row['Phone Number']).strip()
                if phone.endswith('.0'):
                    phone = phone[:-2]

                if not phone.startswith('+91'):
                    phone = '+' + phone

                if phone and phone.lower() != 'nan':

                    msg = (
                        f"Dear Vehicle Owner, your vehicle with plate number {row['Number Plate']} "
                        f"has a challan for {row['Violation']} of ₹{row['Fine Amount']}. Please pay promptly."
                    )
                    try:
                        sns.publish(
                            PhoneNumber=phone,
                            Message=msg,
                            MessageAttributes={
                                'AWS.SNS.SMS.SenderID': {
                                    'DataType': 'String',
                                    'StringValue': SENDER_ID
                                }
                            }
                        )
                        st.success(f"SMS sent to {phone} for Challan ID: {row['Challan ID']}")
                    except Exception as e:
                        st.error(f"Failed to send SMS to {phone}: {e}")
                else:
                    st.warning(f"Missing phone number for Challan ID: {row['Challan ID']}")
except Exception as e:
    st.error(f"Error loading data: {e}")


conn = sqlite3.connect('violations.db')
cursor = conn.cursor()
query = "SELECT * FROM violations"
df = pd.read_sql_query(query, conn)

st.title("Helmet Violations Dashboard")

st.subheader("Recorded Violations")
st.dataframe(df)