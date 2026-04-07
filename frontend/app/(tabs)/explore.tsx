import { useLocalSearchParams, useRouter } from 'expo-router';
import { Pressable, SafeAreaView, ScrollView, StyleSheet, Text, View } from 'react-native';

const progressItems = [
  { label: '오늘 학습 수', value: '4개 단어' },
  { label: '평균 정확도', value: '87%' },
  { label: '연속 학습', value: '7일' },
];

const recentWords = ['사과', '우유', '토끼', '바나나'];

function getText(value: string | string[] | undefined, fallback: string) {
  if (Array.isArray(value)) {
    return value[0] ?? fallback;
  }

  return value ?? fallback;
}

export default function ParentDashboardScreen() {
  const router = useRouter();
  const params = useLocalSearchParams<{
    childName?: string | string[];
    age?: string | string[];
    parentEmail?: string | string[];
  }>();

  const handleGoBack = () => {
    if (router.canGoBack()) {
      router.back();
      return;
    }

    router.replace('/welcome');
  };

  const childName = getText(params.childName, 'Mina');
  const age = getText(params.age, '6');
  const parentEmail = getText(params.parentEmail, 'parent@suksuk.app');

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView contentContainerStyle={styles.container}>
        <Pressable style={styles.backButton} onPress={handleGoBack}>
          <Text style={styles.backButtonText}>← 뒤로가기</Text>
        </Pressable>

        <Text style={styles.title}>부모 대시보드</Text>
        <Text style={styles.subtitle}>{childName}의 발음 연습 현황을 확인해보세요.</Text>

        <View style={styles.profileCard}>
          <Text style={styles.name}>{childName}</Text>
          <Text style={styles.meta}>나이 {age}세 · 초급 단계</Text>
          <Text style={styles.parent}>부모 계정: {parentEmail}</Text>
        </View>

        <View style={styles.sectionCard}>
          <Text style={styles.sectionTitle}>한눈에 보기</Text>
          {progressItems.map((item) => (
            <View key={item.label} style={styles.infoRow}>
              <Text style={styles.infoLabel}>{item.label}</Text>
              <Text style={styles.infoValue}>{item.value}</Text>
            </View>
          ))}
        </View>

        <View style={styles.sectionCard}>
          <Text style={styles.sectionTitle}>최근 연습 단어</Text>
          <View style={styles.badgesWrap}>
            {recentWords.map((word) => (
              <View key={word} style={styles.badge}>
                <Text style={styles.badgeText}>{word}</Text>
              </View>
            ))}
          </View>
        </View>

        <View style={styles.sectionCard}>
          <Text style={styles.sectionTitle}>짧은 코멘트</Text>
          <Text style={styles.sectionText}>• 오늘도 끝까지 집중해서 연습했어요. 정말 잘하고 있어요.</Text>
          <Text style={styles.sectionText}>• 칭찬을 들으면 자신감이 더 올라가는 모습이 보여요.</Text>
        </View>

        <View style={styles.sectionCard}>
          <Text style={styles.sectionTitle}>오늘 발음 피드백</Text>
          <Text style={styles.sectionText}>• “사과”는 “과” 부분을 조금 더 또렷하게 연습하면 좋아요.</Text>
          <Text style={styles.sectionText}>• “토끼”는 마지막 “끼” 소리를 살짝 더 힘 있게 말하면 좋아요.</Text>
          <Text style={styles.sectionText}>• 전체적으로 아주 잘 따라 하고 있어요. 계속 응원해주세요.</Text>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: '#F7FFF8',
  },
  container: {
    padding: 20,
    gap: 14,
    paddingBottom: 28,
  },
  backButton: {
    alignSelf: 'flex-start',
    paddingVertical: 4,
  },
  backButtonText: {
    color: '#6C4CE4',
    fontSize: 14,
    fontWeight: '700',
  },
  title: {
    fontSize: 28,
    fontWeight: '900',
    color: '#1F2544',
  },
  subtitle: {
    fontSize: 15,
    lineHeight: 22,
    color: '#5C677D',
  },
  profileCard: {
    backgroundColor: '#DDF8E8',
    borderRadius: 20,
    padding: 18,
    gap: 4,
  },
  name: {
    fontSize: 24,
    fontWeight: '900',
    color: '#1F2544',
  },
  meta: {
    fontSize: 15,
    color: '#48614A',
  },
  parent: {
    fontSize: 13,
    color: '#48614A',
  },
  sectionCard: {
    backgroundColor: '#FFFFFF',
    borderRadius: 18,
    padding: 16,
    gap: 10,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '800',
    color: '#1F2544',
  },
  infoRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 2,
  },
  infoLabel: {
    fontSize: 14,
    color: '#5C677D',
  },
  infoValue: {
    fontSize: 14,
    fontWeight: '800',
    color: '#1F2544',
  },
  sectionText: {
    fontSize: 14,
    color: '#5C677D',
    lineHeight: 20,
  },
  badgesWrap: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  badge: {
    backgroundColor: '#FFF3C4',
    borderRadius: 999,
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  badgeText: {
    fontWeight: '700',
    color: '#5B3E00',
  },
});
