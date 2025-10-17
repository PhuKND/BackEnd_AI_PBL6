package com.khanhlms.medical_store.configuration;

import com.khanhlms.medical_store.entities.RoleEntity;
import com.khanhlms.medical_store.entities.UserEntity;
import com.khanhlms.medical_store.enums.RoleEnums;
import com.khanhlms.medical_store.repositories.RoleRepository;
import com.khanhlms.medical_store.repositories.UserRepository;
import lombok.AccessLevel;
import lombok.RequiredArgsConstructor;
import lombok.experimental.FieldDefaults;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.ApplicationRunner;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.crypto.password.PasswordEncoder;

import java.util.HashSet;
import java.util.Set;

@Configuration
@RequiredArgsConstructor
@FieldDefaults(level = AccessLevel.PRIVATE, makeFinal = true)
@Slf4j
public class ApplicationInitConfig {
    RoleRepository roleRepository;
    UserRepository userRepository;
    PasswordEncoder passwordEncoder;
    @Bean
    ApplicationRunner initApplicationRunner() {
        return args -> {
            createRole();
            createAdimAccount();
        };
    }

    private void createAdimAccount() {
        if (this.userRepository.findByUsername("admin").isEmpty()) {
            Set<RoleEntity> roles = new HashSet<>(this.roleRepository.findAll());
            this.userRepository.save(UserEntity.builder()
                    .username("admin")
                    .password(passwordEncoder.encode("12345678"))
                    .roles(roles)
                    .isActive(true)
                    .build());
        }
        log.info("Adim account has been created with username: admin, password: 12345678. Pleas change it! ");
    }

    private void createRole() {
        roles(RoleEnums.ADMIN.name());
        roles(RoleEnums.USER.name());
    }

    private void roles(String roles) {
        if (this.roleRepository.findByName(roles).isEmpty()) {
            this.roleRepository.save(RoleEntity.builder()
                    .name(roles)
                    .build());
        }
    }
}
