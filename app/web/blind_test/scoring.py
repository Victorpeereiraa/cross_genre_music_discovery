import streamlit as st
from supabase import create_client


@st.cache_resource
def get_db():
    return create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_SECRET_KEY"]
    )


def save_vote(vote):
    return (
        get_db()
        .table("blind_votes")
        .insert(vote)
        .execute()
    )
