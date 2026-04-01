# Databricks notebook source
# MAGIC %md
# MAGIC # Test — Toast Retail API Access
# MAGIC
# MAGIC Checks whether your current Toast API credentials have access to the
# MAGIC Retail API endpoints used by the inventory pipeline.
# MAGIC
# MAGIC **Run this before attempting a backfill.** It will tell you clearly
# MAGIC whether your credentials work or need the `retail.inventory:read` scope added.

# COMMAND ----------

import requests
import datetime

TOAST_AUTH_URL     = "https://ws.toasttab.com/authentication/v1/authentication/login"
TOAST_INV_HIST_URL = "https://ws.toasttab.com/retail/v1/inventoryHistory/search"

TOAST_CLIENT_ID       = dbutils.secrets.get(scope="toast_api", key="toast_client_id")
TOAST_CLIENT_SECRET   = dbutils.secrets.get(scope="toast_api", key="toast_client_secret")
TOAST_RESTAURANT_GUID = dbutils.secrets.get(scope="toast_api", key="restaurant_guid")

print("✓ Secrets loaded")

# COMMAND ----------

# MAGIC %md ## Step 1 — Authenticate

# COMMAND ----------

auth_resp = requests.post(
    TOAST_AUTH_URL,
    json={
        "clientId":       TOAST_CLIENT_ID,
        "clientSecret":   TOAST_CLIENT_SECRET,
        "userAccessType": "TOAST_MACHINE_CLIENT"
    },
    headers={"Content-Type": "application/json"},
    timeout=30
)

if auth_resp.status_code == 200:
    token = auth_resp.json()["token"]["accessToken"]
    print("✅  PASS — Authentication succeeded")
else:
    print(f"❌  FAIL — Authentication failed: HTTP {auth_resp.status_code}")
    print(auth_resp.text[:500])
    raise RuntimeError("Cannot proceed — fix authentication before testing API access.")

# COMMAND ----------

# MAGIC %md ## Step 2 — Test Retail API (Inventory History)
# MAGIC
# MAGIC Makes a minimal search request for a one-day window.
# MAGIC We don't care about the data — we just need to know if the credentials are authorized.

# COMMAND ----------

yesterday = (datetime.date.today() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
today     = datetime.date.today().strftime("%Y-%m-%d")

inv_resp = requests.post(
    TOAST_INV_HIST_URL,
    headers={
        "Authorization":                f"Bearer {token}",
        "Toast-Restaurant-External-ID": TOAST_RESTAURANT_GUID,
        "Content-Type":                 "application/json"
    },
    json={"updatedDateRange": {"startDate": yesterday, "endDate": today}},
    timeout=30
)

print(f"HTTP Status: {inv_resp.status_code}")
print()

if inv_resp.status_code == 200:
    data = inv_resp.json().get("data", [])
    print(f"✅  PASS — Retail API access confirmed")
    print(f"   Returned {len(data)} inventory history event(s) for {yesterday}")
    print()
    print("   ➜ You are ready to run the backfill.")

elif inv_resp.status_code == 401:
    print("❌  FAIL — 401 Unauthorized")
    print()
    print("   Your credentials authenticated successfully but are not authorized")
    print("   for the Retail API. You need the 'retail.inventory:read' scope added.")
    print()
    print("   Next step: contact your Toast rep or open a Toast support ticket")
    print("   requesting that 'retail.inventory:read' be enabled for client ID:")
    print(f"   {TOAST_CLIENT_ID[:8]}...")

elif inv_resp.status_code == 403:
    print("❌  FAIL — 403 Forbidden")
    print()
    print("   Your credentials are recognized but lack permission for this endpoint.")
    print("   Same resolution as 401 — request 'retail.inventory:read' scope from Toast.")

else:
    print(f"⚠️  UNEXPECTED — HTTP {inv_resp.status_code}")
    print(inv_resp.text[:500])
