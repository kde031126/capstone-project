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

import { submitParentLogin } from '@/lib/api';

export default function LoginScreen() {
  const router = useRouter();
  const [parentEmail, setParentEmail] = useState('');
  const [childName, setChildName] = useState('Mina');
  const [age, setAge] = useState('6');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleContinue = async () => {
    const trimmedEmail = parentEmail.trim() || 'parent@suksuk.app';
    const trimmedChildName = childName.trim() || 'Mina';
    const parsedAge = Number.parseInt(age.trim() || '6', 10);
    const childAge = Number.isNaN(parsedAge) ? 6 : parsedAge;

    try {
      setIsSubmitting(true);

      await submitParentLogin({
        parentEmail: trimmedEmail,
        childName: trimmedChildName,
        age: childAge,
      });

      router.replace({
        pathname: '/(tabs)',
        params: {
          childName: trimmedChildName,
          age: String(childAge),
          parentEmail: trimmedEmail,
        },
      });
    } catch (error) {
      Alert.alert(
        'Backend connection error',
        error instanceof Error ? error.message : 'Unable to connect to the API server.'
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
          <Text style={styles.label}>Parent email</Text>
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

          <Text style={styles.label}>Age</Text>
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
  safeArea: {
    flex: 1,
    backgroundColor: '#F5F7FF',
  },
  container: {
    padding: 24,
    justifyContent: 'center',
    flexGrow: 1,
  },
  backButton: {
    alignSelf: 'flex-start',
    paddingVertical: 8,
    marginBottom: 6,
  },
  backButtonText: {
    color: '#6C4CE4',
    fontWeight: '700',
    fontSize: 14,
  },
  title: {
    fontSize: 30,
    fontWeight: '800',
    color: '#1F2544',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 15,
    lineHeight: 22,
    color: '#5C677D',
    marginBottom: 20,
  },
  card: {
    backgroundColor: '#FFFFFF',
    borderRadius: 22,
    padding: 18,
    gap: 10,
  },
  label: {
    fontSize: 14,
    fontWeight: '700',
    color: '#33415C',
    marginTop: 6,
  },
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
  primaryButton: {
    backgroundColor: '#6C4CE4',
    borderRadius: 16,
    paddingVertical: 15,
    alignItems: 'center',
    marginTop: 12,
  },
  primaryButtonDisabled: {
    opacity: 0.7,
  },
  primaryButtonText: {
    color: '#FFFFFF',
    fontSize: 15,
    fontWeight: '800',
  },
  secondaryButton: {
    alignItems: 'center',
    paddingTop: 10,
  },
  secondaryButtonText: {
    color: '#6C4CE4',
    fontWeight: '700',
  },
});
