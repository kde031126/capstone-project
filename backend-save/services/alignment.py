import numpy as np

def align_phonemes(target_phonemes: str, child_phonemes: str):
    """
    편집 거리(Levenshtein Distance) 알고리즘과 역추적을 사용하여 
    음소 단위의 오류(대치, 탈락, 첨가)를 상세 분석합니다.
    """
    # 공백으로 구분된 음소를 리스트로 변환
    target = target_phonemes.split() if " " in target_phonemes.strip() else list(target_phonemes.replace(" ", ""))
    child = child_phonemes.split() if " " in child_phonemes.strip() else list(child_phonemes.replace(" ", ""))
    
    n, m = len(target), len(child)
    
    # 1. DP 테이블 초기화 (비용 계산용)
    # 행: target(정답), 열: child(아동 발화)
    dp = np.zeros((n + 1, m + 1), dtype=int)
    
    for i in range(n + 1): dp[i][0] = i # Deletion 비용
    for j in range(m + 1): dp[0][j] = j # Insertion 비용
    
    # 2. DP 테이블 채우기
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if target[i-1] == child[j-1]:
                dp[i][j] = dp[i-1][j-1] # 일치
            else:
                dp[i][j] = min(
                    dp[i-1][j],    # Deletion (탈락)
                    dp[i][j-1],    # Insertion (첨가)
                    dp[i-1][j-1]   # Substitution (대치/교체)
                ) + 1

    # 3. 역추적 (Backtracking) - 어느 지점에서 에러가 났는지 확인
    errors = []
    i, j = n, m
    
    while i > 0 or j > 0:
        # 1) 일치하는 경우
        if i > 0 and j > 0 and target[i-1] == child[j-1]:
            i -= 1
            j -= 1
        
        # 2) 대치 (Substitution / 교체)
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            errors.append({
                "type": "substitution",
                "target": target[i-1],
                "error": child[j-1],
                "pos": i - 1  # 정답 음소 기준 위치
            })
            i -= 1
            j -= 1
            
        # 3) 탈락 (Deletion / 정답엔 있는데 아이 발음엔 없음)
        elif i > 0 and (j == 0 or dp[i][j] == dp[i-1][j] + 1):
            errors.append({
                "type": "deletion",
                "target": target[i-1],
                "error": None,
                "pos": i - 1
            })
            i -= 1
            
        # 4) 첨가 (Insertion / 정답엔 없는데 아이가 추가로 발음함)
        elif j > 0 and (i == 0 or dp[i][j] == dp[i][j-1] + 1):
            errors.append({
                "type": "insertion",
                "target": None,
                "error": child[j-1],
                "pos": i # 삽입된 위치
            })
            j -= 1

    # 역추적은 뒤에서부터 수행하므로 다시 정렬
    errors.reverse()
    return errors