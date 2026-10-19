import sys
sys.path.insert(0, r"c:\Users\CHAMILA\Desktop\New folder\Dashboard")
import server
try:
    d = server.build_dashboard()
    print("BUILD OK")
    print("Summary:", d["summary"])
    print()
    for i, s in enumerate(d["services"][:6]):
        print(f"[{i}] {s['id']}: risk={s['risk']} cpu={s['cpu_probability']} mem={s['memory_probability']}")
        print(f"     mem_alarm={s['memory_alarm']} cpu_alarm={s['cpu_alarm']} state={s['state']}")
        print(f"     features={list(s['feature_contributions'].keys())[:4]} rca={str(s['rca_narrative'])[:60]}")
        print(f"     steps={len(s['sre_runbook_steps'])} mitigation_keys={list(s['mitigation_executed'].keys())}")
        print(f"     action={s['action']}")
        print()
except Exception as e:
    import traceback
    print("ERROR:", e)
    traceback.print_exc()
