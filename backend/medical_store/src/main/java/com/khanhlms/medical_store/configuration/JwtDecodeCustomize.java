package com.khanhlms.medical_store.configuration;

import com.khanhlms.medical_store.services.AuthenticationService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Lazy;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.security.oauth2.jose.jws.MacAlgorithm;
import org.springframework.security.oauth2.jwt.Jwt;
import org.springframework.security.oauth2.jwt.JwtDecoder;
import org.springframework.security.oauth2.jwt.JwtException;
import org.springframework.security.oauth2.jwt.NimbusJwtDecoder;

import javax.crypto.spec.SecretKeySpec;
import java.util.Objects;

@Configuration
public class JwtDecodeCustomize implements JwtDecoder {

    @Value("${app.security.secret}")
    private String SECRET;
    @Autowired
    @Lazy
    private AuthenticationService authenticationService;

    private NimbusJwtDecoder nimbusJwtDecoder = null;
    @Override
    public Jwt decode(String token) throws JwtException {
        try {
            boolean introspectToken = this.authenticationService.introspectToken(token);
            if ( !introspectToken) {
                throw new JwtException("Invalid token");
            }
        }
        catch (JwtException e) {
            throw new JwtException(e.getMessage());
        }
        if( Objects.isNull(nimbusJwtDecoder)) {
            SecretKeySpec secretKeySpec = new  SecretKeySpec(SECRET.getBytes(), "HmacSHA512");
            nimbusJwtDecoder = NimbusJwtDecoder.withSecretKey(secretKeySpec)
                    .macAlgorithm(MacAlgorithm.HS512)
                    .build();
        }
        return nimbusJwtDecoder.decode(token);
    }
}
