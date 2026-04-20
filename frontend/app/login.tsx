import { useRouter } from 'expo-router';
import { useState } from 'react';
import {
    Alert,
    Pressable,
    SafeAreaView,
    ScrollView,
    StyleSheet,
    Text,
    TextInput,
    View,
} from 'react-native';

// auth.ts에서 수정된 loginOrCreateUser를 가져옵니다.
import { loginOrCreateUser } from '@/api/auth';

export default function LoginScreen() {
  const router = useRouter();
  
  // 입력 상태 관리
  const [parentEmail, setParentEmail] = useState('');
  const [childName, setChildName] = useState('Mina');
  const [age, setAge] = useState('6');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleContinue = async () => {
    // 1. 데이터 전처리
    const trimmedEmail = parentEmail.trim() || 'test@suksuk.app';
    const trimmedChildName = childName.trim() || 'Mina';
    
    // 2. 백엔드 스키마(birth_year)에 맞게 나이를 출생연도로 변환
    // 2026년 기준 6살이면 2020년생으로 계산
    const currentYear = new Date().getFullYear();
    const inputAge = Number.parseInt(age.trim() || '6', 10);
    const calculatedBirthYear = currentYear - inputAge;

    try {
      setIsSubmitting(true);

      // 3. 백엔드 UserCreate 규격에 100% 맞춰서 전송
      const userData = await loginOrCreateUser({
        user_name: trimmedChildName,      // 스키마 필드명 일치
        birth_year: calculatedBirthYear,  // 스키마 필드명 일치 (int)
        guardian_name: trimmedEmail,      // 스키마 필드명 일치 (보호자 이메일을 이름으로 활용)
        firebase_uid: `temp_${trimmedEmail}`, // 스키마 필드명 일치 (Optional)
      });

      console.log("✅ 서버 응답 성공:", userData);

      // 4. 성공 시 다음 화면으로 이동 (서버에서 준 실제 user_id 전달)
      router.replace({
        pathname: '/(tabs)',
        params: { 
          userId: String(userData.user_id),
          childName: userData.user_name 
        },
      });
    } catch (error) {
      console.error("❌ 통신 에러 상세:", error);
      Alert.alert(
        'Backend connection error',
        error instanceof Error ? error.message : '데이터 형식이 맞지 않거나 서버에 연결할 수 없습니다.'
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleGoBack = () => {
    if (router.canGoBack()) {
      router.back();
      return;
    }
    router.replace('/welcome');
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView contentContainerStyle={styles.container} keyboardShouldPersistTaps="handled">
        <Pressable style={styles.backButton} onPress={handleGoBack}>
          <Text style={styles.backButtonText}>← 뒤로가기</Text>
        </Pressable>

        <Text style={styles.title}>Parent login</Text>
        <Text style={styles.subtitle}>
          Create a quick child profile so lessons can match the right age and level.
        </Text>

        <View style={styles.card}>
          <Text style={styles.label}>Parent email (Guardian)</Text>
          <TextInput
            value={parentEmail}
            onChangeText={setParentEmail}
            style={styles.input}
            placeholder="parent@email.com"
            placeholderTextColor="#8A8FA3"
            keyboardType="email-address"
            autoCapitalize="none"
          />

          <Text style={styles.label}>Child name</Text>
          <TextInput
            value={childName}
            onChangeText={setChildName}
            style={styles.input}
            placeholder="Mina"
            placeholderTextColor="#8A8FA3"
          />

          <Text style={styles.label}>Age (Calculated to Birth Year)</Text>
          <TextInput
            value={age}
            onChangeText={setAge}
            style={styles.input}
            placeholder="6"
            placeholderTextColor="#8A8FA3"
            keyboardType="number-pad"
          />

          <Pressable
            style={[styles.primaryButton, isSubmitting && styles.primaryButtonDisabled]}
            onPress={handleContinue}
            disabled={isSubmitting}>
            <Text style={styles.primaryButtonText}>
              {isSubmitting ? 'Connecting to backend...' : 'Continue to child profile'}
            </Text>
          </Pressable>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: '#F5F7FF' },
  container: { padding: 24, justifyContent: 'center', flexGrow: 1 },
  backButton: { alignSelf: 'flex-start', paddingVertical: 8, marginBottom: 6 },
  backButtonText: { color: '#6C4CE4', fontWeight: '700', fontSize: 14 },
  title: { fontSize: 30, fontWeight: '800', color: '#1F2544', marginBottom: 8 },
  subtitle: { fontSize: 15, lineHeight: 22, color: '#5C677D', marginBottom: 20 },
  card: { backgroundColor: '#FFFFFF', borderRadius: 22, padding: 18, gap: 10 },
  label: { fontSize: 14, fontWeight: '700', color: '#33415C', marginTop: 6 },
  input: {
    backgroundColor: '#F7F8FC',
    borderRadius: 14,
    paddingHorizontal: 14,
    paddingVertical: 12,
    fontSize: 15,
    color: '#1F2544',
    borderWidth: 1,
    borderColor: '#E2E8F0',
  },
  primaryButton: { backgroundColor: '#6C4CE4', borderRadius: 16, paddingVertical: 15, alignItems: 'center', marginTop: 12 },
  primaryButtonDisabled: { opacity: 0.7 },
  primaryButtonText: { color: '#FFFFFF', fontSize: 15, fontWeight: '800' },
});