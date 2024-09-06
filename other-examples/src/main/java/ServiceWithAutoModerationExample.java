import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.model.openai.OpenAiModerationModel;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.Moderate;
import dev.langchain4j.service.ModerationException;

import static dev.langchain4j.model.openai.OpenAiChatModelName.GPT_3_5_TURBO;
import static dev.langchain4j.model.openai.OpenAiModerationModelName.TEXT_MODERATION_LATEST;

public class ServiceWithAutoModerationExample {

    interface Chat {

        @Moderate
        String chat(String text);
    }

    public static void main(String[] args) {

        OpenAiModerationModel moderationModel = OpenAiModerationModel.builder()
                .apiKey(ApiKeys.OPENAI_API_KEY)
                .modelName(TEXT_MODERATION_LATEST)
                .build();

        ChatLanguageModel chatModel = OpenAiChatModel.builder()
                .apiKey(ApiKeys.OPENAI_API_KEY)
                .modelName(GPT_3_5_TURBO)
                .build();

        Chat chat = AiServices.builder(Chat.class)
                .chatLanguageModel(chatModel)
                .moderationModel(moderationModel)
                .build();

        try {
            chat.chat("I WILL KILL YOU!!!");
        } catch (ModerationException e) {
            System.out.println(e.getMessage());
            // Text "I WILL KILL YOU!!!" violates content policy
        }
    }
}
