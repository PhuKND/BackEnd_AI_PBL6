package com.khanhlms.medical_store.utills;

import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Component;

@Component
public class AuthenticationUtills {
    public String getUserName(){
        return SecurityContextHolder.getContext().getAuthentication().getName();
    }
}
