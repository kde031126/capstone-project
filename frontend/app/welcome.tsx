import { useRouter } from 'expo-router';
import { Pressable, SafeAreaView, StyleSheet, Text, View } from 'react-native';

export default function WelcomeScreen() {
  const router = useRouter();

  return (
    <SafeAreaView style={styles.safeArea}>
      <View style={styles.container}>
        <View style={styles.topSection}>
          <Text style={styles.smallTitle}>쑥쑥에 오신 것을</Text>
          <Text style={styles.mainTitle}>환영해요</Text>

          <View style={styles.illustrationWrap}>
            <View style={styles.illustrationCircle}>
              <Text style={styles.illustrationEmoji}>🎓📚</Text>
            </View>
            <View style={styles.sparkleOne} />
            <View style={styles.sparkleTwo} />
          </View>

          <Text style={styles.appName}>쑥쑥</Text>
          <Text style={styles.description}>
            아이들이 쉽고 재미있게 따라 말하며 배우는 한국어 발음 연습 앱이에요.
          </Text>
        </View>

        <View style={styles.bottomSection}>
          <View style={styles.dotsRow}>
            <View style={[styles.dot, styles.dotActive]} />
            <View style={styles.dot} />
            <View style={styles.dot} />
          </View>

          <Pressable style={styles.primaryButton} onPress={() => router.push('/login')}>
            <Text style={styles.primaryButtonText}>시작하기</Text>
          </Pressable>
        </View>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: '#FFFFFF',
  },
  container: {
    flex: 1,
    paddingHorizontal: 24,
    paddingTop: 44,
    paddingBottom: 24,
    justifyContent: 'space-between',
  },
  topSection: {
    alignItems: 'flex-start',
  },
  smallTitle: {
    fontSize: 18,
    color: '#7C8294',
    fontWeight: '700',
    marginBottom: 6,
  },
  mainTitle: {
    fontSize: 40,
    lineHeight: 46,
    color: '#22223B',
    fontWeight: '900',
    marginBottom: 34,
    letterSpacing: -0.8,
  },
  illustrationWrap: {
    width: '100%',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 34,
    position: 'relative',
  },
  illustrationCircle: {
    width: 190,
    height: 190,
    borderRadius: 95,
    backgroundColor: '#F4EEFF',
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: '#7C3AED',
    shadowOpacity: 0.08,
    shadowRadius: 14,
    shadowOffset: { width: 0, height: 6 },
    elevation: 4,
  },
  illustrationEmoji: {
    fontSize: 78,
  },
  sparkleOne: {
    position: 'absolute',
    top: 16,
    right: 42,
    width: 16,
    height: 16,
    borderRadius: 8,
    backgroundColor: '#DCCFFF',
  },
  sparkleTwo: {
    position: 'absolute',
    bottom: 22,
    left: 46,
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: '#E9DDFF',
  },
  appName: {
    fontSize: 30,
    fontWeight: '900',
    color: '#22223B',
    textAlign: 'center',
    alignSelf: 'center',
    marginBottom: 10,
    letterSpacing: -0.4,
  },
  description: {
    fontSize: 17,
    lineHeight: 26,
    color: '#6B7280',
    textAlign: 'center',
    paddingHorizontal: 8,
  },
  bottomSection: {
    gap: 20,
  },
  dotsRow: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 8,
  },
  dot: {
    width: 9,
    height: 9,
    borderRadius: 5,
    backgroundColor: '#D1D5DB',
  },
  dotActive: {
    width: 24,
    backgroundColor: '#7C3AED',
  },
  primaryButton: {
    backgroundColor: '#7C3AED',
    borderRadius: 14,
    paddingVertical: 17,
    alignItems: 'center',
  },
  primaryButtonText: {
    color: '#FFFFFF',
    fontSize: 18,
    fontWeight: '900',
  },
});
