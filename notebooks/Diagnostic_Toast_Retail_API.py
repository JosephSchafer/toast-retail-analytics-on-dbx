# Databricks notebook source
# MAGIC %md
# MAGIC # Diagnostic — Toast Retail API Access
# MAGIC
# MAGIC Deep diagnostic for the `retail.inventory:read` scope issue.
# MAGIC Runs 8 tests and prints the full raw response for each so you can give
# MAGIC Toast support the exact error details.
# MAGIC
# MAGIC **Share the full output of this notebook with your Toast PM / support contact.**

# COMMAND ----------

import requests
import json
import datetime
import base64

# ── Load credentials ──────────────────────────────────────────────────────────
TOAST_CLIENT_ID       = dbutils.secrets.get(scope="toast_api", key="toast_client_id")
TOAST_CLIENT_SECRET   = dbutils.secrets.get(scope="toast_api", key="toast_client_secret")
TOAST_RESTAURANT_GUID = dbutils.secrets.get(scope="toast_api", key="restaurant_guid")

TOAST_AUTH_URL          = "https://ws.toasttab.com/authentication/v1/authentication/login"
TOAST_SANDBOX_AUTH_URL  = "https://ws-sandbox-api.eng.toasttab.com/authentication/v1/authentication/login"

# Production host variants
PROD_HOST    = "https://ws.toasttab.com"
# Sandbox host (from Toast sample code — retail API may live here even for prod credentials)
SANDBOX_HOST = "https://ws-sandbox-api.eng.toasttab.com"

# Endpoint variants to try across both hosts
INVENTORY_ENDPOINTS = [
    f"{PROD_HOST}/v1/inventoryHistory/search",
    f"{SANDBOX_HOST}/v1/inventoryHistory/search",
    f"{PROD_HOST}/retail/v1/inventoryHistory/search",
    f"{SANDBOX_HOST}/retail/v1/inventoryHistory/search",
]
PO_ENDPOINTS = [
    f"{PROD_HOST}/v1/purchaseOrders/search",
    f"{SANDBOX_HOST}/v1/purchaseOrders/search",
    f"{PROD_HOST}/retail/v1/purchaseOrders/search",
    f"{SANDBOX_HOST}/retail/v1/purchaseOrders/search",
]

yesterday = (datetime.date.today() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
today     = datetime.date.today().strftime("%Y-%m-%d")

print("=" * 70)
print("TOAST RETAIL API DIAGNOSTIC")
print(f"Run at: {datetime.datetime.utcnow().isoformat()}Z")
print(f"Client ID (first 8): {TOAST_CLIENT_ID[:8]}...")
print(f"Restaurant GUID:      {TOAST_RESTAURANT_GUID}")
print(f"Date window:          {yesterday} → {today}")
print("=" * 70)

# COMMAND ----------

# MAGIC %md ## Test 1 — Authentication

# COMMAND ----------

print("\n── TEST 1: Authentication ─────────────────────────────────────────────")

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

print(f"HTTP Status:  {auth_resp.status_code}")
print(f"Response Headers:")
for k, v in auth_resp.headers.items():
    print(f"  {k}: {v}")
print(f"\nResponse Body (raw):")
print(auth_resp.text[:2000])

if auth_resp.status_code != 200:
    raise RuntimeError(f"AUTH FAILED — HTTP {auth_resp.status_code}. Cannot proceed.")

token_data = auth_resp.json()
TOKEN = token_data["token"]["accessToken"]
print(f"\n✅  Auth succeeded — token obtained")

# COMMAND ----------

# MAGIC %md ## Test 2 — Decode JWT Token Claims

# COMMAND ----------

print("\n── TEST 2: JWT Token Claims ───────────────────────────────────────────")
print("Decoding token payload to inspect granted scopes and expiry...\n")

try:
    # JWT is header.payload.signature — base64url decode the payload
    parts = TOKEN.split(".")
    if len(parts) == 3:
        payload_b64 = parts[1]
        # Pad to multiple of 4 for base64 decoding
        payload_b64 += "=" * (4 - len(payload_b64) % 4)
        payload_bytes = base64.urlsafe_b64decode(payload_b64)
        payload = json.loads(payload_bytes)

        print("JWT Payload claims:")
        for k, v in payload.items():
            if k in ("exp", "iat", "nbf"):
                # Convert epoch to human-readable
                dt = datetime.datetime.utcfromtimestamp(v)
                print(f"  {k}: {v}  ({dt.isoformat()}Z)")
            else:
                print(f"  {k}: {v}")

        # Highlight scope
        scopes = payload.get("scope", payload.get("scp", payload.get("scopes", None)))
        if scopes:
            print(f"\n  *** SCOPES GRANTED: {scopes} ***")
        else:
            print("\n  ⚠️  No 'scope'/'scp'/'scopes' claim found in token payload.")
            print("      Keys present:", list(payload.keys()))
    else:
        print(f"  Token does not look like a JWT (parts: {len(parts)})")
        print(f"  Token prefix: {TOKEN[:50]}...")
except Exception as e:
    print(f"  Could not decode token: {e}")
    print(f"  Token prefix: {TOKEN[:50]}...")

# COMMAND ----------

# MAGIC %md ## Test 3 — Inventory History (all endpoint variants)

# COMMAND ----------

print("\n── TEST 3: Inventory History Endpoint Variants ────────────────────────")

inv_results = {}
for url in INVENTORY_ENDPOINTS:
    print(f"\nTrying: {url}")
    try:
        resp = requests.post(
            url,
            headers={
                "Authorization":                f"Bearer {TOKEN}",
                "Toast-Restaurant-External-ID": TOAST_RESTAURANT_GUID,
                "Content-Type":                 "application/json",
            },
            json={"updatedDateRange": {"startDate": yesterday, "endDate": today}},
            timeout=30
        )
        print(f"  Status: {resp.status_code}")
        print(f"  Body:   {resp.text[:500]}")
        inv_results[url] = resp.status_code
    except Exception as e:
        print(f"  ERROR: {e}")
        inv_results[url] = "ERROR"

# COMMAND ----------

# MAGIC %md ## Test 4 — Purchase Orders (all endpoint variants)

# COMMAND ----------

print("\n── TEST 4: Purchase Orders Endpoint Variants ──────────────────────────")

po_results = {}
for url in PO_ENDPOINTS:
    print(f"\nTrying: {url}")
    try:
        resp = requests.post(
            url,
            headers={
                "Authorization":                f"Bearer {TOKEN}",
                "Toast-Restaurant-External-ID": TOAST_RESTAURANT_GUID,
                "Content-Type":                 "application/json",
            },
            json={"updatedDateRange": {"startDate": yesterday, "endDate": today}},
            timeout=30
        )
        print(f"  Status: {resp.status_code}")
        print(f"  Body:   {resp.text[:500]}")
        po_results[url] = resp.status_code
    except Exception as e:
        print(f"  ERROR: {e}")
        po_results[url] = "ERROR"

# COMMAND ----------

# MAGIC %md ## Test 5 — Wider date window (in case yesterday has no data)

# COMMAND ----------

print("\n── TEST 5: Wider Date Window (last 30 days) ───────────────────────────")
thirty_ago = (datetime.date.today() - datetime.timedelta(days=30)).strftime("%Y-%m-%d")

url = INVENTORY_ENDPOINTS[0]
print(f"Endpoint: {url}")
print(f"Window:   {thirty_ago} → {today}")

resp = requests.post(
    url,
    headers={
        "Authorization":                f"Bearer {TOKEN}",
        "Toast-Restaurant-External-ID": TOAST_RESTAURANT_GUID,
        "Content-Type":                 "application/json",
    },
    json={"updatedDateRange": {"startDate": thirty_ago, "endDate": today}},
    timeout=30
)
print(f"Status: {resp.status_code}")
print(f"Body:   {resp.text[:1000]}")

# COMMAND ----------

# MAGIC %md ## Test 6 — Without restaurant GUID header

# COMMAND ----------

print("\n── TEST 6: Without Toast-Restaurant-External-ID Header ────────────────")
print("(Some endpoints infer it from the token — testing if header is the issue)")

url = INVENTORY_ENDPOINTS[0]
resp = requests.post(
    url,
    headers={
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type":  "application/json",
    },
    json={"updatedDateRange": {"startDate": yesterday, "endDate": today}},
    timeout=30
)
print(f"Status: {resp.status_code}")
print(f"Body:   {resp.text[:500]}")

# COMMAND ----------

# MAGIC %md ## Test 7 — GET by ID (exact pattern from Toast sample code)

# COMMAND ----------

print("\n── TEST 7: GET by ID — exact Toast sample code pattern ────────────────")
print("Toast sample uses: GET /v1/inventoryHistory/{id}")
print("A 404 = endpoint reachable but ID not found (good — means we have access)")
print("A 401/403 = permission denied (scope issue confirmed)")
print("A 200 = unexpected hit on dummy ID\n")

dummy_id = "00000000-0000-0000-0000-000000000000"

for host in [PROD_HOST, SANDBOX_HOST]:
    url = f"{host}/v1/inventoryHistory/{dummy_id}"
    print(f"Trying: {url}")
    try:
        resp = requests.get(
            url,
            headers={
                "Authorization":                f"Bearer {TOKEN}",
                "Toast-Restaurant-External-ID": TOAST_RESTAURANT_GUID,
            },
            timeout=30
        )
        print(f"  Status: {resp.status_code}")
        print(f"  Body:   {resp.text[:300]}")
        if resp.status_code == 404:
            print(f"  ✅  404 = endpoint exists and you have access — dummy ID just not found")
        elif resp.status_code in (401, 403):
            print(f"  ❌  {resp.status_code} = permission denied on this host")
    except Exception as e:
        print(f"  ERROR: {e}")

# COMMAND ----------

# MAGIC %md ## Test 7b — Auth against sandbox host

# COMMAND ----------

print("\n── TEST 7b: Authenticate directly against sandbox host ────────────────")
print("(Checking if the retail API requires its own auth endpoint)")

sandbox_auth_resp = requests.post(
    TOAST_SANDBOX_AUTH_URL,
    json={
        "clientId":       TOAST_CLIENT_ID,
        "clientSecret":   TOAST_CLIENT_SECRET,
        "userAccessType": "TOAST_MACHINE_CLIENT"
    },
    headers={"Content-Type": "application/json"},
    timeout=30
)
print(f"Status: {sandbox_auth_resp.status_code}")
print(f"Body:   {sandbox_auth_resp.text[:500]}")

if sandbox_auth_resp.status_code == 200:
    SANDBOX_TOKEN = sandbox_auth_resp.json()["token"]["accessToken"]
    print("\n✅  Sandbox auth succeeded — testing inventory endpoint with sandbox token")
    resp = requests.get(
        f"{SANDBOX_HOST}/v1/inventoryHistory/{dummy_id}",
        headers={
            "Authorization":                f"Bearer {SANDBOX_TOKEN}",
            "Toast-Restaurant-External-ID": TOAST_RESTAURANT_GUID,
        },
        timeout=30
    )
    print(f"  Inventory GET status: {resp.status_code}")
    print(f"  Body: {resp.text[:300]}")
else:
    print("  Sandbox auth failed — credentials may be production-only")

# COMMAND ----------

# MAGIC %md ## Test 8 — Call a known-working endpoint with same token

# COMMAND ----------

print("\n── TEST 8: Known-working Orders API with same token ───────────────────")
print("(Confirms the token works generally — isolates whether it's a scope issue)")

orders_url = "https://ws.toasttab.com/orders/v2/ordersBulk"
resp = requests.get(
    orders_url,
    headers={
        "Authorization":                f"Bearer {TOKEN}",
        "Toast-Restaurant-External-ID": TOAST_RESTAURANT_GUID,
    },
    params={
        "startDate": f"{yesterday}T00:00:00.000-0500",
        "endDate":   f"{today}T00:00:00.000-0500",
        "pageSize":  1
    },
    timeout=30
)
print(f"Status: {resp.status_code}")
print(f"Body:   {resp.text[:300]}")
if resp.status_code == 200:
    print("  ✅  Orders API works with this token — confirms auth is fine")
    print("      If inventory is 401/403, it is specifically a scope/permission issue")
else:
    print("  ❌  Orders API also failed — token itself may be wrong credential set")

# COMMAND ----------

# MAGIC %md ## Summary

# COMMAND ----------

print()
print("=" * 70)
print("DIAGNOSTIC SUMMARY")
print("=" * 70)
print(f"\nClient ID (first 8): {TOAST_CLIENT_ID[:8]}...")
print(f"Restaurant GUID:      {TOAST_RESTAURANT_GUID}")
print(f"\nInventory History results:")
for url, status in inv_results.items():
    icon = "✅" if status == 200 else "❌"
    print(f"  {icon}  {status}  {url}")
print(f"\nPurchase Orders results:")
for url, status in po_results.items():
    icon = "✅" if status == 200 else "❌"
    print(f"  {icon}  {status}  {url}")
print()
print("WHAT TO SHARE WITH TOAST SUPPORT:")
print("  1. The full output of this notebook")
print(f"  2. Client ID prefix: {TOAST_CLIENT_ID[:8]}...")
print(f"  3. Restaurant GUID: {TOAST_RESTAURANT_GUID}")
print("  4. The JWT claims from Test 2 — especially the 'scope' field")
print("     If 'retail.inventory:read' is missing, the PM needs to enable it")
print("  5. HTTP status codes from Tests 3 & 4 (all endpoint variants)")
print("  6. Test 7 result — 404 means access granted, 401/403 means scope missing")
print("  7. Test 7b — whether sandbox auth works (indicates if host matters)")
print("  8. Test 8 — whether Orders API passes (isolates scope vs credential issue)")
print()
print("KEY QUESTION FOR TOAST PM:")
print("  The Toast sample code uses host: ws-sandbox-api.eng.toasttab.com")
print("  Does the retail API require a different host than ws.toasttab.com?")
print("  And does client ID starting with '6Nmo' have retail.inventory:read enabled?")
print("=" * 70)
