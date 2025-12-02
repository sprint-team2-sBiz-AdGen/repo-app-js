import React, { useState, useMemo } from 'react';
import { View, Text, TouchableOpacity, Image, TextInput, StyleSheet, Alert, ScrollView, ActivityIndicator, Linking, KeyboardAvoidingView, Platform } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { commonStyles as cs } from './_styles';
import { createGenerationJob, gptKorToEng, gptAdCopyEng, gptAdCopyKor } from '../api/feedlyApi'; // <--- Import the NEW function
import { STRATEGIES } from '../constants/strategies';

export default function PhotoAndDescriptionScreen({ route, navigation }) {
  const { strategy } = route.params || {};
  const [imageUri, setImageUri] = useState(null);
  const [description, setDescription] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const strategyLabel = strategy?.label || "스타일 선택";

  const descriptionPlaceholder = useMemo(() => {
    const foundStrategy = STRATEGIES.find(s => s.id === strategy?.id);
    return foundStrategy?.placeholder || "제품이나 서비스에 대한 설명을 자유롭게 입력해 주세요.";
  }, [strategy]);

  const pickImage = async () => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert(
        'Permission Required',
        'Please grant photo library permissions in your settings to continue.',
        [{ text: "Cancel", style: "cancel" }, { text: "Open Settings", onPress: () => Linking.openSettings() }]
      );
      return;
    }

    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: 'Images',
        allowsEditing: true,
        aspect: [4, 5],
        quality: 0.8,
      });

      if (!result.canceled) {
        setImageUri(result.assets[0].uri);
      }
    } catch (error) {
      console.error("Error launching image picker:", error);
      Alert.alert("Error", "Could not open the photo library.");
    }
  };

  const handleGenerate = async () => {
    if (!description.trim()) {
      Alert.alert("입력 필요", "광고에 대한 설명을 입력해 주세요.");
      return;
    }

    setIsLoading(true);

    try {
      //console.log("--- Starting Generation Job ---");
    
      // 1. Call the NEW API endpoint
      const result = await createGenerationJob(imageUri, description);
      
      gptKorToEng(result.job_id)
      gptAdCopyEng(result.job_id)
      gptAdCopyKor(result.job_id)

      // console.log("--- Job Started Successfully ---", result);
      
      // 2. Navigate to the next screen (or show success)
      // For now, we just alert the Job ID. Later we will navigate to a 'Generating' screen.
      // Alert.alert("Success", `Job Started! ID: ${result.job_id}`);
      
      navigation.navigate('GeneratingScreen', { jobId: result.job_id });

    } catch (error) {
      console.error("Generation failed:", error);
      Alert.alert("Error", "Failed to start generation. Check console for details.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <KeyboardAvoidingView
      behavior={Platform.OS === "ios" ? "padding" : "height"}
      style={{ flex: 1 }}
      keyboardVerticalOffset={90}
    >
      <ScrollView style={cs.container} contentContainerStyle={{ flexGrow: 1 }}>
        <Text style={cs.title}>{strategyLabel}</Text>
        <Text style={cs.subtitle}>광고에 사용할 사진과 설명을 입력해 주세요.</Text>

        <TouchableOpacity onPress={pickImage} style={styles.imagePicker}>
          {imageUri ? (
            <Image source={{ uri: imageUri }} style={styles.imagePreview} />
          ) : (
            <View style={styles.imagePlaceholder}>
              <Image
                source={require('../assets/images/icon-add-image.png')}
                style={styles.imagePlaceholderIcon}
              />
              <Text style={styles.imagePlaceholderText}>광고에 사용할 이미지를 업로드해 주세요.</Text>
            </View>
          )}
        </TouchableOpacity>

        <TextInput
          style={styles.textInput}
          placeholder={descriptionPlaceholder}
          placeholderTextColor="#6b7280"
          multiline
          value={description}
          onChangeText={setDescription}
        />

        <View style={{ flex: 1 }} />

        <TouchableOpacity
          style={[cs.primaryButton, (isLoading || !description.trim()) && { backgroundColor: "#d1d5db" }]}
          onPress={handleGenerate}
          disabled={isLoading || !description.trim()}
        >
          {isLoading ? <ActivityIndicator color="#fff" /> : <Text style={cs.primaryButtonText}>광고 생성 시작</Text>}
        </TouchableOpacity>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  imagePicker: { width: '85%', alignSelf: 'center', aspectRatio: 4 / 5, backgroundColor: '#f3f4f6', borderRadius: 12, justifyContent: 'center', alignItems: 'center', marginTop: 20, borderWidth: 1, borderColor: '#e5e7eb' },
  imagePreview: { width: '100%', height: '100%', borderRadius: 12 },
  imagePlaceholder: { justifyContent: 'center', alignItems: 'center', padding: 20 },
  imagePlaceholderIcon: { width: 48, height: 48, marginBottom: 12, opacity: 0.5 },
  imagePlaceholderText: { color: '#6b7280', fontSize: 16, textAlign: 'center' },
  textInput: { width: '100%', minHeight: 120, backgroundColor: '#f3f4f6', borderRadius: 12, padding: 16, fontSize: 16, textAlignVertical: 'top', marginTop: 20, borderWidth: 1, borderColor: '#e5e7eb' },
});
