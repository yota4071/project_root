import json
from datetime import datetime

# ゾーン名を数値に変換
zone_map = {
    "zone_A": 1,
    "zone_B": 2,
    "zone_C": 3,
    "zone_D": 4
}

# ISO形式→UNIX秒（int）へ変換
def iso_to_unix(timestamp_str):
    return int(datetime.fromisoformat(timestamp_str).timestamp())

# 最新のzone_D到達者の軌跡を抽出してCircom用inputに変換
def extract_latest_zone_d(file_path, output_path):
    with open(file_path, 'r') as f:
        all_data = json.load(f)

    latest_person = None
    latest_d_time = None

    for person_id, records in all_data.items():
        for idx, entry in enumerate(records):
            if entry["zone"] == "zone_D":
                ts = datetime.fromisoformat(entry["timestamp"])
                if latest_d_time is None or ts > latest_d_time:
                    latest_d_time = ts
                    latest_person = (person_id, records[:idx + 1])  # A〜Dの軌跡だけ残す

    if latest_person is None:
        print("[!] zone_D に到達した人物がいませんでした。")
        return

    person_id, trajectory = latest_person
    zones = [zone_map[pt["zone"]] for pt in trajectory]
    timestamps = [iso_to_unix(pt["timestamp"]) for pt in trajectory]

    output = {
        "zones": zones,
        "timestamps": timestamps
        # maxDuration は不要になったため削除
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"[✔] {person_id} のデータを Circom 用に変換し保存しました → {output_path}")

if __name__ == "__main__":
    extract_latest_zone_d("data/trajectories.json", "circuits/input.json")