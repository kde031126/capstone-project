import { Link, useRouter } from 'expo-router';
import { Pressable, StyleSheet } from 'react-native';

import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';

export default function ModalScreen() {
  const router = useRouter();

  const handleGoBack = () => {
    if (router.canGoBack()) {
      router.back();
      return;
    }

    router.replace('/welcome');
  };

  return (
    <ThemedView style={styles.container}>
      <Pressable style={styles.backButton} onPress={handleGoBack}>
        <ThemedText type="link">← 뒤로가기</ThemedText>
      </Pressable>
      <ThemedText type="title">This is a modal</ThemedText>
      <Link href="/" dismissTo style={styles.link}>
        <ThemedText type="link">Go to home screen</ThemedText>
      </Link>
    </ThemedView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
  },
  backButton: {
    position: 'absolute',
    top: 56,
    left: 20,
  },
  link: {
    marginTop: 15,
    paddingVertical: 15,
  },
});
