# 데이터 매핑 테이블
region_labels = {
    '서울': 0, '부산': 1, '대구': 2, '인천': 3, '광주': 4, '대전': 5, '울산': 6, 
    '세종': 7, '경기남부': 8, '경기북부': 9, '강원영서': 10, '강원영동': 11, '충북': 12, 
    '충남': 13, '전북': 14, '전남': 15, '경북': 16, '경남': 17, '제주': 18
}
sex_labels = {'남성': 0, '여성': 1, '상관 없음': 2}
c_canwalk_labels = {'지원 가능': 0, '지원 불가능': 1, '둘 다 케어 가능': 2}
p_canwalk_labels = {'걸을 수 없음': 0, '걸을 수 있음': 1}
spot_label = {'병원': 0, '집': 1, '둘 다': 2}
smoking_labels = {'비흡연': 0, '흡연': 1, '상관 없음': 2}
symptom_labels = {
    '치매': 0, '섬망': 1, '욕창': 2, '하반신 마비': 3, '상반신 마비': 4, '전신 마비': 5, '와상 환자': 6, 
    '기저귀 케어': 7, '의식 없음': 8, '석션': 9, '피딩': 10, '소변줄': 11, '장루': 12, 
    '야간 집중 돌봄': 13, '전염성': 14, '파킨슨': 15, '정신질환': 16, '투석': 17, '재활': 18
}

# 매핑 함수들
def map_region(region):
    regions = region.split(',')
    a = [str(region_labels[r]) for r in regions if r in region_labels]
    return ','.join(a)

def map_sex(sex):
    sexx = sex.split(',')
    a = [str(sex_labels[r]) for r in sexx if r in sex_labels]
    return ','.join(a)

def map_spot(spot):
    spott = spot.split(',')
    a = [str(spot_label[r]) for r in spott if r in spot_label]
    return ','.join(a)

def map_canwalk(canwalk):
    canwalkk = canwalk.split(',')
    a = [str(c_canwalk_labels[r]) for r in canwalkk if r in c_canwalk_labels]
    return ','.join(a)

def map_symtoms(symptom):
    symptomm = symptom.split(',')
    a = [str(symptom_labels[r]) for r in symptomm if r in symptom_labels]
    return ','.join(a)

def map_prefersex(prefersex):
    prefersexx = prefersex.split(',')
    a = [str(sex_labels[r]) for r in prefersexx if r in sex_labels]
    return ','.join(a)

def map_smoking(smoking):
    smokingg = smoking.split(',')
    a = [str(smoking_labels[r]) for r in smokingg if r in smoking_labels]
    return ','.join(a)

def map_pwalk(pwalk):
    pwalkk = pwalk.split(',')
    a = [str(p_canwalk_labels[r]) for r in pwalkk if r in p_canwalk_labels]
    return ','.join(a)

def check_spot_match(row):
    if row['spot_x'] == row['spot_y']:
        return True
    if int(row['spot_y']) == 2:
        return True
    return False

def check_gender_match(row):
    if (int(row['prefersex_x']) == int(row['sex_y'])) or (int(row['prefersex_x']) == 2):
        if (int(row['prefersex_y']) == int(row['sex_x'])) or (int(row['prefersex_y']) == 2):
            return True
        else:
            return False
    else:
        return False

def check_canwalk_match(row):
    if int(row['canwalk_x']) == int(row['canwalk_y']):
        return True
    if int(row['canwalk_y']) == 2:
        return True
    return False

def check_smoking_match(row):
    if int(row['smoking_x']) == 2:
        return True
    if int(row['smoking_x']) == int(row['smoking_y']):
        return True
    return False

def calculate_matching_rate1(row):
    matching_features = 0
    if row["region_match"] == 1:
        matching_features += 2
    if row["spot_match"] == 1:
        matching_features += 2
    if row["gender_match"] == 1:
        matching_features += 2
    if row["canwalk_match"] == 1:
        matching_features += 2
    if row["smoking_match"] == 1:
        matching_features += 1
    if row["symptom_match_score"] == 1:
        matching_features += 2
    if 0.5 <= row["symptom_match_score"] < 1:
        matching_features += 1
    if row["symptom_match_score"] < 0.5:
        matching_features += row["symptom_match_score"]
    if row["date_overlap"] == 1:
        matching_features += 2
    return matching_features / 13

def calculate_matching_rate2(row):
    if row.get('hard_matching_rate', 0) == 1:
        return 99.9  
    else:
        return row.get('tab_matching_rate', 0) * 100
