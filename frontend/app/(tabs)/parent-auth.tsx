import { useRouter } from 'expo-router';
import { useState } from 'react';
import {
  KeyboardAvoidingView,
  Platform,
  Pressable,
  SafeAreaView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';

export default function ParentAuthScreen() {
  const router = useRouter();
  const [email, setEmail] = useState('');
  
  const [isLoggingIn, setIsLoggingIn] = useState(false);

    const handleAccessDashboard = () => {
    // 1. 로딩 상태 시작
    setIsLoggingIn(true);

    // 2. 2초 정도 딜레이를 준 후 이동 (보안 확인하는 느낌 연출)
    setTimeout(() => {
        setIsLoggingIn(false); // 로딩 종료
        router.push('/explore'); 
    }, 3000); 
    };  

  const handleGoBack = () => {
    router.back();
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <KeyboardAvoidingView 
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={styles.container}
      >
        <Pressable style={styles.backButton} onPress={handleGoBack}>
          <Text style={styles.backButtonText}>← 돌아가기</Text>
        </Pressable>

        <View style={styles.content}>
          <View style={styles.iconCircle}>
            <Text style={styles.icon}>🔒</Text>
          </View>
          
          <Text style={styles.title}>부모 전용 공간입니다</Text>
          <Text style={styles.subtitle}>
            아이의 학습 현황을 확인하시려면{"\n"}보호자의 이메일을 입력해 주세요.
          </Text>

          <View style={styles.inputCard}>
            <Text style={styles.label}>부모 이메일</Text>
            <TextInput
              style={styles.input}
              placeholder="example@email.com"
              placeholderTextColor="#A0AEC0"
              keyboardType="email-address"
              autoCapitalize="none"
              value={email}
              onChangeText={setEmail}
            />
            
            <Pressable 
  style={({ pressed }) => [
    styles.button,
    (pressed || isLoggingIn) && styles.buttonPressed,
    isLoggingIn && { backgroundColor: '#A0AEC0' } // 로딩 중일 때 버튼 색상을 회색으로 변경
  ]} 
  onPress={handleAccessDashboard}
  disabled={isLoggingIn} // 🔥 중요: 로딩 중일 때 버튼 중복 클릭 방지
>
  <Text style={styles.buttonText}>
    {/* 🔥 상태에 따라 텍스트 변경 */}
    {isLoggingIn ? '대시보드 불러오는 중...' : '대시보드 확인하기'}
  </Text>
</Pressable>
          </View>

        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: '#F7FAFC',
  },
  container: {
    flex: 1,
    padding: 24,
  },
  backButton: {
    paddingVertical: 12,
  },
  backButtonText: {
    color: '#718096',
    fontSize: 16,
    fontWeight: '600',
  },
  content: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingBottom: 40,
  },
  iconCircle: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: '#EDF2F7',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 24,
  },
  icon: {
    fontSize: 40,
  },
  title: {
    fontSize: 24,
    fontWeight: '900',
    color: '#2D3748',
    marginBottom: 12,
  },
  subtitle: {
    fontSize: 16,
    color: '#718096',
    textAlign: 'center',
    lineHeight: 24,
    marginBottom: 32,
  },
  inputCard: {
    width: '100%',
    backgroundColor: '#FFFFFF',
    borderRadius: 24,
    padding: 24,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.05,
    shadowRadius: 12,
    elevation: 3,
  },
  label: {
    fontSize: 14,
    fontWeight: '700',
    color: '#4A5568',
    marginBottom: 8,
    marginLeft: 4,
  },
  input: {
    backgroundColor: '#F7FAFC',
    borderRadius: 16,
    paddingHorizontal: 16,
    paddingVertical: 14,
    fontSize: 16,
    borderWidth: 1,
    borderColor: '#E2E8F0',
    marginBottom: 20,
  },
  button: {
    backgroundColor: '#63D400',
    borderRadius: 16,
    paddingVertical: 16,
    alignItems: 'center',
  },
  buttonPressed: {
    opacity: 0.9,
    transform: [{ scale: 0.98 }],
  },
  buttonText: {
    color: '#FFFFFF',
    fontSize: 17,
    fontWeight: '800',
  },
  footerText: {
    marginTop: 24,
    fontSize: 13,
    color: '#A0AEC0',
    textAlign: 'center',
  },
});