package com.khanhlms.medical_store.configuration;


import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import com.cloudinary.Cloudinary;

@Configuration
public class CloudinaryConfig {

    @Value("${app.clodinary.url}")
    private String cloudinaryUrl;

    @Bean
    public Cloudinary cloudinary() {
        return new Cloudinary(cloudinaryUrl);
    }
}