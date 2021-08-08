import React, { Component, useState } from 'react';
import { Text, View, StyleSheet, Image, Button } from 'react-native';
import { Constants } from 'expo';
import { ImageEditor } from "expo-image-editor";
import * as ImagePicker from "expo-image-picker";

const App = () => {
  const [imageUri, setImageUri] = useState(undefined);
  const [croppedUri, setCroppedUri] = useState(undefined);
  const [editorVisible, setEditorVisible] = useState(false);
  const [aspectLock, setAspectLock] = useState(false);
  
  const selectPhoto = async () => {
    // Get the permission to access the camera roll
    const response = await ImagePicker.requestCameraRollPermissionsAsync();
    // If they said yes then launch the image picker
    if (response.granted) {
      const pickerResult = await ImagePicker.launchImageLibraryAsync();
      // Check they didn't cancel the picking
      if (!pickerResult.cancelled) {
        launchEditor(pickerResult.uri);
      }
    } else {
      // If not then alert the user they need to enable it
      Alert.alert(
        "Please enable camera roll permissions for this app in your settings."
      );
    }
  };

  const launchEditor = (uri: string) => {
    // Then set the image uri
    setImageUri(uri);
    // And set the image editor to be visible
    setEditorVisible(true);
  };

  return (
    <View>
      <Image
        style={{ height: 300, width: 300 }}
        source={{ uri: imageUri }}
      />
      <Button title="Select Photo" onPress={() => selectPhoto()} />
      <ImageEditor
        visible={editorVisible}
        onCloseEditor={() => setEditorVisible(false)}
        imageUri={imageUri}
        fixedCropAspectRatio={16 / 9}
        lockAspectRatio={aspectLock}
        minimumCropDimensions={{
          width: 100,
          height: 100,
        }}
        onEditingComplete={(result) => {
          setCroppedUri(result.uri);
        }}
        mode="full"
      />
      <Button color="#ff5c5c" title="Take Picture"  />
      
  </View>
  );
}

export default App;

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#ecf0f1',
  },
  paragraph: {
    margin: 24,
    padding: 20,
    fontSize: 18,
    fontWeight: 'bold',
    textAlign: 'center',
    color: '#34495e',
  },
});