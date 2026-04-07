import { useLocalSearchParams, useRouter } from 'expo-router';
import { useEffect, useState } from 'react';
import { Pressable, SafeAreaView, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';

const levels = ['초급', '중급', '고급'] as const;

function getText(value: string | string[] | undefined, fallback: string) {
  if (Array.isArray(value)) {
    return value[0] ?? fallback;
  }

  return value ?? fallback;
}

export default function SettingsScreen() {
  const router = useRouter();
  const params = useLocalSearchParams<{
    childName?: string | string[];
    age?: string | string[];
    level?: string | string[];
  }>();

  const childName = getText(params.childName, 'Mina');
  const age = getText(params.age, '6');
  const initialLevel = getText(params.level, '초급');

  const [editableName, setEditableName] = useState(childName);
  const [editableAge, setEditableAge] = useState(age);
  const [selectedLevel, setSelectedLevel] = useState(initialLevel);
  const [savedMessage, setSavedMessage] = useState('');

  useEffect(() => {
    setEditableName(childName);
    setEditableAge(age);
    setSelectedLevel(initialLevel);
  }, [childName, age, initialLevel]);

  const handleGoBack = () => {
    if (router.canGoBack()) {
      router.back();
      return;
    }

    router.replace('/welcome');
  };

  const handleSave = () => {
    const nextName = editableName.trim() || 'Mina';
    const nextAge = editableAge.replace(/[^0-9]/g, '').slice(0, 2) || '6';

    setEditableName(nextName);
    setEditableAge(nextAge);
    setSavedMessage(`${nextName}의 정보가 저장되었어요.`);

    router.replace({
      pathname: '/(tabs)/settings',
      params: {
        childName: nextName,
        age: nextAge,
        level: selectedLevel,
      },
    });
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView contentContainerStyle={styles.container} keyboardShouldPersistTaps="handled">
        <Pressable style={styles.backButton} onPress={handleGoBack}>
          <Text style={styles.backButtonText}>← 뒤로가기</Text>
        </Pressable>

        <Text style={styles.title}>아이 정보설정</Text>
        <Text style={styles.subtitle}>이름, 나이, 학습 단계를 바꿔 아이에게 맞는 연습을 준비해요.</Text>

        <View style={styles.card}>
          <Text style={styles.label}>아이 이름</Text>
          <TextInput
            value={editableName}
            onChangeText={(text) => {
              setEditableName(text);
              setSavedMessage('');
            }}
            style={styles.input}
            placeholder="이름 입력"
            placeholderTextColor="#8A8FA3"
          />

          <Text style={styles.label}>나이</Text>
          <TextInput
            value={editableAge}
            onChangeText={(text) => {
              setEditableAge(text.replace(/[^0-9]/g, ''));
              setSavedMessage('');
            }}
            style={styles.input}
            placeholder="나이 입력"
            placeholderTextColor="#8A8FA3"
            keyboardType="number-pad"
          />

          <Text style={styles.label}>학습 단계</Text>
          <View style={styles.levelRow}>
            {levels.map((level) => {
              const isActive = selectedLevel === level;

              return (
                <Pressable
                  key={level}
                  style={[styles.levelChip, isActive && styles.activeChip]}
                  onPress={() => {
                    setSelectedLevel(level);
                    setSavedMessage('');
                  }}>
                  <Text style={[styles.levelChipText, isActive && styles.activeChipText]}>{level}</Text>
                </Pressable>
              );
            })}
          </View>

          <Pressable style={styles.saveButton} onPress={handleSave}>
            <Text style={styles.saveButtonText}>정보 저장</Text>
          </Pressable>

          {savedMessage ? <Text style={styles.savedMessage}>{savedMessage}</Text> : null}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: '#FFFDF6',
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
  card: {
    backgroundColor: '#FFFFFF',
    borderRadius: 18,
    padding: 16,
    gap: 10,
  },
  label: {
    fontSize: 14,
    fontWeight: '700',
    color: '#33415C',
    marginTop: 4,
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
  levelRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    marginTop: 4,
  },
  levelChip: {
    backgroundColor: '#F3EEFF',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 999,
  },
  activeChip: {
    backgroundColor: '#6C4CE4',
  },
  levelChipText: {
    color: '#5B38D1',
    fontWeight: '700',
  },
  activeChipText: {
    color: '#FFFFFF',
    fontWeight: '800',
  },
  saveButton: {
    backgroundColor: '#6C4CE4',
    borderRadius: 14,
    paddingVertical: 14,
    alignItems: 'center',
    marginTop: 10,
  },
  saveButtonText: {
    color: '#FFFFFF',
    fontSize: 15,
    fontWeight: '800',
  },
  savedMessage: {
    marginTop: 4,
    color: '#2F855A',
    fontSize: 14,
    fontWeight: '700',
    textAlign: 'center',
  },
});
