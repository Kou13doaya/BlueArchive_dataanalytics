import os
import shutil
import pandas as pd

def main():
    file_path = "rank_data/rank_data_total_assault_90_20260707_0120.parquet"
    backup_path = file_path + ".bak"
    
    # バックアップの作成
    if not os.path.exists(backup_path):
        shutil.copyfile(file_path, backup_path)
        print(f"[INFO] Backup created at {backup_path}")
    else:
        print(f"[INFO] Backup already exists at {backup_path}")

    # データの読み込み
    df = pd.read_parquet(backup_path)
    print(f"Original shape: {df.shape}")

    # インデックス（rank）順にソートしてリストに変換
    df_sorted = df.sort_index()
    scores_list = df_sorted['score'].tolist()
    ranks_list = df_sorted.index.tolist()

    # 挿入データ (1-indexedの順位, スコア)
    # 昇順にソートされていることを確認
    insert_data = [
        (23, 53726280),
        (32, 53721864),
        (39, 53717159),
        (193, 53709621),
        (314, 53708904),
        (1137, 53289288),
        (1143, 53286792),
        (1882, 53132519),
        (1912, 53128775),
        (2454, 53070887),
        (2462, 53070216),
        (2640, 53059848),
        (2766, 53057929),
        (3075, 53054759),
        (3426, 53048423),
        (4266, 53033640),
        (4352, 53032776),
        (4585, 53029608),
        (4817, 52981802),
        (6278, 52534633),
        (6503, 52410891),
        (6530, 52400999),
        (7130, 40064320),
        (7282, 40063840),
        (7509, 40063600),
        (7605, 40063441),
        (19127, 39997201),
        (19667, 39980720),
        (20298, 39906721),
        (20897, 39867280)
    ]
    
    # 挿入処理
    # 順位が小さい順に処理するため、insert_data を rank で昇順ソートしておく
    insert_data.sort(key=lambda x: x[0])
    
    # scores_list は 0-indexed なので、rank位に差し込むには index = rank - 1 に insert する。
    # 昇順ループで順番に insert していけば、既存の要素は自然に後ろにずれていく。
    for rank, score in insert_data:
        idx = rank - 1
        # リストのサイズを超えている場合は末尾に追加
        if idx >= len(scores_list):
            scores_list.append(score)
        else:
            scores_list.insert(idx, score)

    # 新しいDataFrameの作成
    # 順位は1から始まる連番にする
    new_ranks = list(range(1, len(scores_list) + 1))
    new_df = pd.DataFrame({'score': scores_list}, index=new_ranks)
    new_df.index.name = 'rank'
    
    print(f"New shape: {new_df.shape}")
    
    # 差し込んだ箇所の確認
    print("\nVerification of inserted points:")
    for rank, expected_score in insert_data:
        actual_score = new_df.loc[rank, 'score']
        status = "OK" if actual_score == expected_score else "NG"
        print(f"Rank {rank}: Expected={expected_score}, Actual={actual_score} ({status})")

    # ファイルに書き戻し
    new_df.to_parquet(file_path)
    print(f"[SUCCESS] Updated file saved to {file_path}")

if __name__ == '__main__':
    main()
