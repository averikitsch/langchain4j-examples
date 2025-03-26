
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import dev.langchain4j.engine.EmbeddingStoreConfig;
import dev.langchain4j.engine.PostgresEngine;
import dev.langchain4j.engine.MetadataColumn;
import dev.langchain4j.store.embedding.cloudsql.PostgresEmbeddingStore;
import java.sql.SQLException;


import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

public class PostgresEmbeddingStoreWithMetadataExample {

    private static final String TABLE_NAME = "POSTGRES_EMBEDDING_TEST_TABLE";
    private static final Integer VECTOR_SIZE = 384;

    public static void main(String[] args) throws SQLException {

        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        PostgresEngine engine = new PostgresEngine.Builder()
                .projectId(System.getenv("POSTGRES_PROJECT_ID"))
                .region(System.getenv("REGION"))
                .instance(System.getenv("POSTGRES_INSTANCE"))
                .database(System.getenv("POSTGRES_DB"))
                .user(System.getenv("POSTGRES_USER"))
                .password(System.getenv("POSTGRES_PASS"))
                .ipType("public")
                .build();

        List<MetadataColumn> metadataColumns = new ArrayList<>();
        metadataColumns.add(new MetadataColumn("userId", "uuid", true));

        EmbeddingStoreConfig embeddingStoreConfig = new EmbeddingStoreConfig.Builder(TABLE_NAME, VECTOR_SIZE)
                .metadataColumns(metadataColumns)
                .storeMetadata(true)
                .overwriteExisting(true)
                .build();

        engine.initVectorStoreTable(embeddingStoreConfig);

        List<String> metaColumnNames =
                metadataColumns.stream().map(c -> c.getName()).collect(Collectors.toList());

        PostgresEmbeddingStore embeddingStore = new PostgresEmbeddingStore.Builder(engine, TABLE_NAME)
                .metadataColumns(metaColumnNames)
                .build();


        Metadata metadata1 = new Metadata();
        metadata1.put("userId", UUID.randomUUID());
        TextSegment segment1 = TextSegment.from("I like turtles.", metadata1);
        Embedding embedding1 = embeddingModel.embed(segment1).content();
        embeddingStore.add(embedding1, segment1);

        Metadata metadata2 = new Metadata(); 
        metadata2.put("userId", UUID.randomUUID());
        TextSegment segment2 = TextSegment.from("All right!. You're a great zombie", metadata2);
        Embedding embedding2 = embeddingModel.embed(segment2).content();
        embeddingStore.add(embedding2, segment2);

        Embedding queryEmbedding = embeddingModel.embed("What is your favourite animal?").content();
        EmbeddingSearchRequest searchRequest = EmbeddingSearchRequest.builder()
                .queryEmbedding(queryEmbedding)
                .maxResults(1)
                .build();
        EmbeddingSearchResult<TextSegment> searchResult = embeddingStore.search(searchRequest);
        EmbeddingMatch<TextSegment> embeddingMatch = searchResult.matches().get(0);
        
        System.out.println("Unfiltered match:");
        System.out.println(embeddingMatch.score());
        System.out.println(embeddingMatch.embedded().text());

        // Search embedding store with filter
        Filter onlyForUser1 = metadataKey("userId").isEqualTo(user1);
        EmbeddingSearchRequest embeddingSearchRequest1 = EmbeddingSearchRequest.builder()
                        .queryEmbedding(queryEmbedding).filter(onlyForUser1).build();
        EmbeddingSearchResult<TextSegment> embeddingSearchResult1 =
                        embeddingStore.search(embeddingSearchRequest1);
        EmbeddingMatch<TextSegment> embeddingMatch1 =
                        embeddingSearchResult1.matches().get(0);
        
        System.out.println("Filtered match:");
        System.out.println(embeddingMatch1.score());
        System.out.println(embeddingMatch1.embedded().text());

        engine.close();
    }
}
