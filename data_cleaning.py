"""
Data Cleaning Script for CatFood.csv
=====================================
แบบสำรวจความคิดเห็นเกี่ยวกับบรรจุภัณฑ์อาหารแมวสำเร็จรูปชนิดเม็ด
- ลบแถว Brief/คำอธิบาย (4 แถวแรก) และ git merge conflict markers
- ตั้งชื่อคอลัมน์ให้สั้นกระชับ
- ลบแถวว่าง และแถวที่ถูกคัดกรองออก (ไม่เคยซื้ออาหารแมว)
- แปลง Likert scale เป็นตัวเลข
- แปลงคอลัมน์อายุเป็นตัวเลข
- บันทึกไฟล์ที่ทำความสะอาดแล้ว
"""

import pandas as pd
import numpy as np
import os

# ============================================================
# 1. อ่านไฟล์ CSV - ข้ามแถว Brief (4 แถวแรก)
# ============================================================
print("=" * 60)
print("1. กำลังอ่านไฟล์ CatFood.csv ...")
print("=" * 60)

raw_df = pd.read_csv(
    r"d:\Work\Final\CatFood.csv",
    skiprows=4,       # ข้าม 4 แถวแรก (Brief + git conflict)
    encoding="utf-8",
)

print(f"   ขนาดข้อมูลดิบ: {raw_df.shape[0]} แถว x {raw_df.shape[1]} คอลัมน์")

# ============================================================
# 2. ตั้งชื่อคอลัมน์ใหม่ให้สั้นกระชับ
# ============================================================
print("\n" + "=" * 60)
print("2. ตั้งชื่อคอลัมน์ใหม่ ...")
print("=" * 60)

new_columns = [
    "timestamp",
    "experience",           # เคยซื้ออาหารแมวหรือไม่
    "cat_meaning",          # แมวมีความหมายอย่างไร
    "cat_breed",            # พันธุ์แมว
    "current_brand",        # แบรนด์ที่ซื้อปัจจุบัน
    "factor_natural",       # วัตถุดิบธรรมชาติ
    "factor_imported",      # วัตถุดิบนำเข้า
    "factor_taste",         # รสชาติกลมกล่อม
    "factor_foreign",       # ผลิตภัณฑ์ต่างประเทศ
    "factor_brand_fame",    # แบรนด์มีชื่อเสียง
    "packaging_influence",  # บรรจุภัณฑ์มีผลต่อการซื้อ
    "image_preference",     # ชอบภาพแบบใด
    "pkg_premium",          # บรรจุภัณฑ์ดูพรีเมียม
    "pkg_cat_image",        # มีภาพแมว
    "pkg_kibble_image",     # มีภาพอาหารเม็ด
    "pkg_ingredient_image", # มีภาพวัตถุดิบ
    "pkg_eco_friendly",     # เป็นมิตรสิ่งแวดล้อม
    "pkg_origin_symbol",    # สัญลักษณ์แหล่งผลิต
    "pkg_benefit_symbol",   # สัญลักษณ์ประโยชน์
    "pkg_guarantee",        # การการันตี
    "memorable_brand",      # แบรนด์ที่โดดเด่น
    "desired_addition",     # อยากเติมอะไรในบรรจุภัณฑ์
]

# Option 1-10 แต่ละ option มี 5 คำถาม
option_aspects = ["want_buy", "standout", "premium", "taste", "suit_me"]

for opt_num in range(1, 11):
    for aspect in option_aspects:
        new_columns.append(f"opt{opt_num}_{aspect}")

# คอลัมน์สุดท้าย
new_columns.extend(["top3_choices", "age", "gender", "marital_status"])

# ตรวจสอบจำนวนคอลัมน์ตรงกัน
if len(new_columns) != raw_df.shape[1]:
    print(f"   ⚠️ จำนวนคอลัมน์ไม่ตรง: ตั้งชื่อ {len(new_columns)} vs ข้อมูล {raw_df.shape[1]}")
    print(f"   ใช้ชื่อคอลัมน์เดิมแทน")
else:
    raw_df.columns = new_columns
    print(f"   ✅ ตั้งชื่อคอลัมน์ใหม่สำเร็จ ({len(new_columns)} คอลัมน์)")

# ============================================================
# 3. ลบแถวว่าง + แถวที่มี git merge conflict
# ============================================================
print("\n" + "=" * 60)
print("3. ลบแถวว่างและ git merge conflict ...")
print("=" * 60)

before_count = len(raw_df)

# ลบแถวที่ทุกคอลัมน์เป็น NaN หรือว่าง
raw_df = raw_df.dropna(how="all")

# ลบแถวที่มี git merge conflict markers
conflict_mask = raw_df.apply(
    lambda row: row.astype(str).str.contains(r"^[<>=]{7}", regex=True).any(),
    axis=1,
)
raw_df = raw_df[~conflict_mask]

after_count = len(raw_df)
print(f"   ลบแถวว่าง/conflict: {before_count - after_count} แถว")
print(f"   เหลือ: {after_count} แถว")

# ============================================================
# 4. ลบแถวที่ถูกคัดกรองออก (ไม่เคยซื้ออาหารแมว)
# ============================================================
print("\n" + "=" * 60)
print("4. ลบแถวผู้ตอบที่ไม่เคยซื้ออาหารแมว ...")
print("=" * 60)

before_count = len(raw_df)

# คัดเฉพาะคนที่ตอบ "เคย"
screened_out = raw_df[raw_df["experience"] == "ไม่เคย"]
raw_df = raw_df[raw_df["experience"] == "เคย"].copy()

after_count = len(raw_df)
print(f"   ลบผู้ตอบที่ไม่เคยซื้อ: {before_count - after_count} คน")
print(f"   เหลือผู้ตอบที่ผ่านคัดกรอง: {after_count} คน")

# ============================================================
# 5. แปลง Likert scale เป็นตัวเลข
# ============================================================
print("\n" + "=" * 60)
print("5. แปลง Likert scale เป็นตัวเลข ...")
print("=" * 60)

# 5.1 Importance scale (5 ระดับ) — ใช้กับคอลัมน์ factor_* และ pkg_*
importance_map = {
    "มากที่สุด": 5,
    "มาก": 4,
    "ปานกลาง": 3,
    "น้อย": 2,
    "น้อยที่สุด": 1,
}

importance_cols = [
    "factor_natural", "factor_imported", "factor_taste",
    "factor_foreign", "factor_brand_fame",
    "pkg_premium", "pkg_cat_image", "pkg_kibble_image",
    "pkg_ingredient_image", "pkg_eco_friendly",
    "pkg_origin_symbol", "pkg_benefit_symbol", "pkg_guarantee",
]

for col in importance_cols:
    if col in raw_df.columns:
        raw_df[col] = raw_df[col].map(importance_map)

print(f"   ✅ แปลง Importance scale ({len(importance_cols)} คอลัมน์)")

# 5.2 Agreement scale (5 ระดับ) — ใช้กับคอลัมน์ opt*
agreement_map = {
    "เห็นด้วยที่สุด": 5,
    "เห็นด้วย": 4,
    "เฉยๆ": 3,
    "ไม่เห็นด้วย": 2,
    "ไม่เห็นด้วยเลย": 1,
}

option_cols = [c for c in raw_df.columns if c.startswith("opt")]

for col in option_cols:
    if col in raw_df.columns:
        raw_df[col] = raw_df[col].map(agreement_map)

print(f"   ✅ แปลง Agreement scale ({len(option_cols)} คอลัมน์)")

# 5.3 packaging_influence (มีผล/ไม่มีผล) → 1/0
raw_df["packaging_influence"] = raw_df["packaging_influence"].map({
    "มีผล": 1,
    "ไม่มีผล": 0,
})
print("   ✅ แปลง packaging_influence เป็น 1/0")

# ============================================================
# 6. แปลงคอลัมน์อายุเป็นตัวเลข (ค่ากลาง)
# ============================================================
print("\n" + "=" * 60)
print("6. แปลงช่วงอายุ ...")
print("=" * 60)

age_map = {
    "ต่ำกว่า 20 ปี": "< 20",
    "20-29ปี": "20-29",
    "30-39ปี": "30-39",
    "40-49ปี": "40-49",
    "50ปี ขึ้นไป": "50+",
}
raw_df["age"] = raw_df["age"].map(age_map)

print("   ✅ ทำความสะอาดช่วงอายุ")
print(f"   การกระจาย:\n{raw_df['age'].value_counts().to_string()}")

# ============================================================
# 7. ทำความสะอาด text columns (ตัดช่องว่างหัว-ท้าย)
# ============================================================
print("\n" + "=" * 60)
print("7. ทำความสะอาดคอลัมน์ข้อความ ...")
print("=" * 60)

text_cols = [
    "cat_meaning", "cat_breed", "current_brand",
    "image_preference", "memorable_brand", "desired_addition",
    "top3_choices", "gender", "marital_status",
]

for col in text_cols:
    if col in raw_df.columns:
        raw_df[col] = raw_df[col].astype(str).str.strip()
        raw_df[col] = raw_df[col].replace({"nan": np.nan, "None": np.nan, "": np.nan, "-": np.nan})

print(f"   ✅ ทำความสะอาด {len(text_cols)} คอลัมน์ข้อความ")

# ============================================================
# 8. ลบคอลัมน์ที่ไม่จำเป็น (experience ซ้ำ เพราะกรองเหลือ 'เคย' แล้ว)
# ============================================================
print("\n" + "=" * 60)
print("8. จัดการคอลัมน์สุดท้าย ...")
print("=" * 60)

raw_df = raw_df.drop(columns=["experience"])
print("   ✅ ลบคอลัมน์ 'experience' (ทุกแถวเป็น 'เคย' แล้ว)")

# Reset index
raw_df = raw_df.reset_index(drop=True)

# ============================================================
# 9. ตรวจสอบ Missing Values
# ============================================================
print("\n" + "=" * 60)
print("9. สรุป Missing Values ...")
print("=" * 60)

missing = raw_df.isnull().sum()
missing_pct = (missing / len(raw_df) * 100).round(1)
missing_summary = pd.DataFrame({
    "missing_count": missing,
    "missing_pct": missing_pct,
}).query("missing_count > 0").sort_values("missing_pct", ascending=False)

if len(missing_summary) > 0:
    print(missing_summary.to_string())
else:
    print("   ✅ ไม่มี Missing Values!")

# ============================================================
# 10. สรุปและบันทึกไฟล์
# ============================================================
print("\n" + "=" * 60)
print("10. สรุปข้อมูลหลังทำความสะอาด")
print("=" * 60)

print(f"\n   📊 ขนาดข้อมูลสุดท้าย: {raw_df.shape[0]} แถว x {raw_df.shape[1]} คอลัมน์")
print(f"   📋 ประเภทข้อมูล:")
print(f"      - ตัวเลข (numeric): {raw_df.select_dtypes(include=[np.number]).shape[1]} คอลัมน์")
print(f"      - ข้อความ (object): {raw_df.select_dtypes(include=['object']).shape[1]} คอลัมน์")

print(f"\n   📋 คอลัมน์ทั้งหมด:")
for i, col in enumerate(raw_df.columns, 1):
    dtype = raw_df[col].dtype
    print(f"      {i:2d}. {col:<25s} ({dtype})")

# บันทึกไฟล์
output_path = r"d:\Work\Final\data\CatFood_cleaned.csv"
raw_df.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"\n   ✅ บันทึกไฟล์: {output_path}")

# บันทึกสรุป
print("\n" + "=" * 60)
print("  การทำความสะอาดข้อมูลเสร็จสมบูรณ์!")
print("=" * 60)

# แสดง 5 แถวแรก
print("\n--- ตัวอย่างข้อมูล 5 แถวแรก ---")
print(raw_df.head().to_string())

# แสดง descriptive statistics ของคอลัมน์ตัวเลข
print("\n--- สถิติเชิงพรรณนา (Likert columns) ---")
numeric_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()
if numeric_cols:
    print(raw_df[numeric_cols].describe().round(2).to_string())
