package com.khanhlms.medical_store.utills;

import com.cloudinary.Cloudinary;
import com.cloudinary.utils.ObjectUtils;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.util.Map;

@Component
@RequiredArgsConstructor
public class CloudinaryUtils {
    private final Cloudinary cloudinary;
    public String uploadImageCloddy(MultipartFile file) {
        try {
            Map upload = cloudinary.uploader().upload(
                    file.getBytes(),
                    ObjectUtils.emptyMap()
            );
            return (String) upload.get("url");
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    public String uploadImageCloddy(File file) {
        try {
            Map<?, ?> upload = cloudinary.uploader().upload(
                    file,
                    ObjectUtils.emptyMap()
            );
            return (String) upload.get("url");
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }
}
